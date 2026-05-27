"""
sistema_electrolineras.py
=========================
Sistema de consola para simular y analizar una red de electrolineras
en el área metropolitana de Bucaramanga (Colombia), desarrollado como
Proyecto Integrador de la Facultad de Ingeniería en Ciencia de Datos - UIS.

Combina:
  - Grafos geográficos (NetworkX / OSMnx)
  - Algoritmos clásicos: Dijkstra, Floyd-Warshall, QuickSort, BFS, DFS,
    Búsqueda Binaria
  - Machine Learning: Random Forest (clasificación de zonas Voronoi) y
    K-Means (sugerencia de nuevas ubicaciones)
  - Visualización interactiva con Folium
  - Exportación de reportes a JSON, CSV, TXT y XLSX

Uso:
    python sistema_electrolineras.py
"""

import os, json, csv, random, time, math, warnings
import heapq
import collections
import numpy as np
import pandas as pd
import networkx as nx
import folium

# Directorio base donde se guardan los archivos de salida
BASE_DIR = r"C:\Users\luisp\OneDrive\Escritorio\proyectofinal"

from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

warnings.filterwarnings("ignore")

# ── Dependencias opcionales ────────────────────────────────────────────────────

try:
    import osmnx as ox
    ox.settings.log_console = False
    OSMNX_OK = True   # OSMnx disponible: permite cargar red vial real
except Exception:
    OSMNX_OK = False  # Fallback al grafo sintético

try:
    import openpyxl
    XLSX_OK = True    # Permite exportar estadísticas a Excel
except Exception:
    XLSX_OK = False

# ── Datos del sistema ──────────────────────────────────────────────────────────

# 8 electrolineras reales ubicadas en Bucaramanga y Piedecuesta.
# Cada entrada registra id, nombre, coordenadas, tipo de establecimiento
# y un contador de recargas que se actualiza en la simulación.
ELECTROLINERAS = [
    {"id": 0, "nombre": "Homecenter",             "lat": 7.1157, "lon": -73.1210, "tipo": "Comercial",    "recargas": 0},
    {"id": 1, "nombre": "CC Quinta Etapa",        "lat": 7.1151, "lon": -73.1087, "tipo": "C.Comercial",  "recargas": 0},
    {"id": 2, "nombre": "CC Cacique",             "lat": 7.0996, "lon": -73.1073, "tipo": "C.Comercial",  "recargas": 0},
    {"id": 3, "nombre": "CC Canaveral",           "lat": 7.0707, "lon": -73.1069, "tipo": "C.Comercial",  "recargas": 0},
    {"id": 4, "nombre": "Terpel Piedecuesta",     "lat": 7.0358, "lon": -73.0714, "tipo": "Gasolinera",   "recargas": 0},
    {"id": 5, "nombre": "Exito La Rosita",        "lat": 7.1141, "lon": -73.1229, "tipo": "Supermercado", "recargas": 0},
    {"id": 6, "nombre": "CC La Florida",          "lat": 7.0700, "lon": -73.1048, "tipo": "C.Comercial",  "recargas": 0},
    {"id": 7, "nombre": "Promotores del Oriente", "lat": 7.0857, "lon": -73.1647, "tipo": "Industrial",   "recargas": 0},
]

# 10 puntos de referencia (sedes universitarias y centros) usados como
# origen y destino en la simulación de recorridos. IDs 10-19.
PUNTOS_REF = [
    {"id": 10, "nombre": "UIS Campus Central",   "lat": 7.1408, "lon": -73.1209},
    {"id": 11, "nombre": "UIS Campus Florida",   "lat": 7.0617, "lon": -73.0885},
    {"id": 12, "nombre": "UIS Guatiguara",       "lat": 6.9943, "lon": -73.0657},
    {"id": 13, "nombre": "UIS Bucarica",         "lat": 7.1197, "lon": -73.1233},
    {"id": 14, "nombre": "CENFER",               "lat": 7.0824, "lon": -73.1543},
    {"id": 15, "nombre": "UNAB",                 "lat": 7.1163, "lon": -73.1050},
    {"id": 16, "nombre": "UTS",                  "lat": 7.1050, "lon": -73.1234},
    {"id": 17, "nombre": "UPB",                  "lat": 7.0381, "lon": -73.0717},
    {"id": 18, "nombre": "PTAR Rio Frio",        "lat": 7.0656, "lon": -73.1280},
    {"id": 19, "nombre": "Sede Catay",           "lat": 6.9760, "lon": -73.0415},
]

# 2 vehículos eléctricos con especificaciones reales.
# El estado de batería y los contadores se actualizan durante la simulación.
VEHICULOS = [
    {"id": 0, "nombre": "Mercedes-Benz EQS 450+", "gama": "Alta",
     "bateria_kwh": 107.8, "rango_km": 770,
     "bateria_actual_pct": 100.0, "recargas_totales": 0, "km_recorridos": 0.0},
    {"id": 1, "nombre": "Renault Zoe R110", "gama": "Baja",
     "bateria_kwh": 52.0, "rango_km": 395,
     "bateria_actual_pct": 100.0, "recargas_totales": 0, "km_recorridos": 0.0},
]

# ── Estado global ──────────────────────────────────────────────────────────────

G                 = None  # Grafo activo (NetworkX); se inicializa al arrancar
modelo_ml         = None  # Random Forest entrenado (módulo 8)
scaler_ml         = None  # StandardScaler asociado al modelo ML
registro_recargas = []    # Historial de cada recarga ocurrida en simulaciones


# ══════════════════════════════════════════════════════════════════════════════
# UTILIDADES GENERALES
# ══════════════════════════════════════════════════════════════════════════════

def limpiar():
    """Limpia la pantalla de la terminal (compatible con Windows y Unix)."""
    os.system("cls" if os.name == "nt" else "clear")


def separador(titulo=""):
    """
    Imprime una línea separadora de 62 caracteres.
    Si se proporciona un título, lo muestra centrado debajo de la línea.
    """
    print("\n" + "=" * 62)
    if titulo:
        print(f"  {titulo}")
        print("=" * 62)


def validar_entero(msg, minv, maxv):
    """
    Solicita un entero al usuario y lo valida en el rango [minv, maxv].

    Repite la solicitud hasta recibir una entrada válida.

    Args:
        msg  (str): Mensaje que se muestra al pedir el dato.
        minv (int): Valor mínimo aceptado (inclusivo).
        maxv (int): Valor máximo aceptado (inclusivo).

    Returns:
        int: El entero validado ingresado por el usuario.
    """
    while True:
        entrada = input(msg).strip()
        if not entrada.lstrip("-").isdigit():
            print("  ⚠ Solo números enteros.")
            continue
        v = int(entrada)
        if v < minv or v > maxv:
            print(f"  ⚠ Valor fuera de rango [{minv} - {maxv}].")
            continue
        return v


def validar_texto(msg):
    """
    Solicita una cadena de texto no vacía al usuario.

    Acepta nombres con números y espacios (ej. 'CC Cacique 2', 'Etapa 3').

    Args:
        msg (str): Mensaje que se muestra al pedir el dato.

    Returns:
        str: La cadena ingresada, sin espacios al inicio o al final.
    """
    while True:
        entrada = input(msg).strip()
        if not entrada:
            print("  ⚠ El campo no puede estar vacío.")
            continue
        return entrada


def haversine(lat1, lon1, lat2, lon2):
    """
    Calcula la distancia geodésica entre dos puntos geográficos.

    Usa la fórmula de Haversine sobre la esfera terrestre (R = 6371 km).
    Es la métrica de distancia usada en todo el sistema.

    Args:
        lat1, lon1 (float): Coordenadas del punto de origen.
        lat2, lon2 (float): Coordenadas del punto de destino.

    Returns:
        float: Distancia en kilómetros entre los dos puntos.
    """
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(lat1))
         * math.cos(math.radians(lat2))
         * math.sin(dlon / 2) ** 2)
    return R * 2 * math.asin(math.sqrt(a))


def nombre_por_id(nid):
    """
    Devuelve el nombre de un nodo (electrolinera o punto de referencia) por su ID.

    Args:
        nid (int): ID del nodo buscado.

    Returns:
        str: Nombre del nodo, o la representación en cadena del ID si no se encuentra.
    """
    todos = ELECTROLINERAS + PUNTOS_REF
    for n in todos:
        if n["id"] == nid:
            return n["nombre"]
    return str(nid)


# ══════════════════════════════════════════════════════════════════════════════
# MÓDULO 1 — CONSTRUCCIÓN Y CARGA DEL GRAFO
# ══════════════════════════════════════════════════════════════════════════════

def construir_grafo_sintetico():
    """
    Construye un grafo no dirigido (NetworkX) conectando cada nodo con sus
    4 vecinos más cercanos, usando la distancia Haversine como peso de arista.

    Nodos: las 8 electrolineras (IDs 0-7) y los 10 puntos de referencia (IDs 10-19).
    Se usa como fallback cuando OSMnx no está disponible o el usuario lo prefiere.

    Modifica la variable global G.
    """
    global G
    G = nx.Graph()
    todos = ELECTROLINERAS + PUNTOS_REF

    # Agregar nodos con atributos geográficos y de tipo
    for nodo in todos:
        tipo = "electrolinera" if nodo["id"] < 10 else "referencia"
        G.add_node(nodo["id"], nombre=nodo["nombre"],
                   lat=nodo["lat"], lon=nodo["lon"], tipo=tipo)

    # Conectar cada nodo con sus 4 vecinos más cercanos por distancia Haversine
    for i in range(len(todos)):
        distancias = []
        for j in range(len(todos)):
            if i != j:
                d = haversine(todos[i]["lat"], todos[i]["lon"],
                              todos[j]["lat"], todos[j]["lon"])
                distancias.append((d, todos[j]["id"]))
        distancias.sort()
        for k in range(min(4, len(distancias))):
            G.add_edge(todos[i]["id"], distancias[k][1],
                       weight=round(distancias[k][0], 4))


def cargar_grafo_osmnx():
    """
    Intenta descargar la red vial real de Bucaramanga usando OSMnx.

    Si tiene éxito, asigna el grafo real a G y agrega a cada nodo del sistema
    el nodo OSM más cercano ('osm_node'). Si falla por cualquier razón,
    llama a construir_grafo_sintetico() como fallback.

    Modifica la variable global G.

    Returns:
        bool: True si OSMnx se cargó correctamente, False si se usó el sintético.
    """
    global G
    try:
        print("  Descargando red vial (OSMnx)...")
        G_osm = ox.graph_from_place("Bucaramanga, Colombia", network_type="drive")
        G = ox.convert.to_undirected(G_osm)
        todos = ELECTROLINERAS + PUNTOS_REF
        for nodo in todos:
            nn = ox.distance.nearest_nodes(G, nodo["lon"], nodo["lat"])
            nodo["osm_node"] = nn
        print("  ✓ Grafo OSMnx cargado.")
        return True
    except Exception as e:
        print(f"  ✗ OSMnx falló: {e}")
        print("  → Usando grafo sintético...")
        construir_grafo_sintetico()
        return False


def mod_cargar_grafo():
    """
    Módulo 1 del menú: permite al usuario elegir entre OSMnx y el grafo sintético.

    Si OSMnx no está instalado, construye directamente el grafo sintético.
    Muestra el número de nodos, aristas y si el grafo es conexo.
    """
    separador("MÓDULO 1 — CARGA DEL GRAFO")
    if OSMNX_OK:
        resp = input("  ¿Usar OSMnx (red vial real)? (s/n): ").strip().lower()
        if resp == "s":
            cargar_grafo_osmnx()
        else:
            construir_grafo_sintetico()
            print("  ✓ Grafo sintético construido.")
    else:
        construir_grafo_sintetico()
        print("  ✓ Grafo sintético construido (OSMnx no disponible).")
    print(f"  Nodos: {G.number_of_nodes()} | Aristas: {G.number_of_edges()}")
    print(f"  Grafo conexo: {'Sí' if nx.is_connected(G) else 'No'}")
    input("\n  Enter para continuar...")


# ══════════════════════════════════════════════════════════════════════════════
# MÓDULO 2 — VISUALIZACIÓN DE DATOS
# ══════════════════════════════════════════════════════════════════════════════

def mod_ver_nodos():
    """
    Módulo 2 del menú: muestra en consola las tablas de electrolineras,
    puntos de referencia y vehículos con todos sus atributos.
    """
    separador("MÓDULO 2 — ELECTROLINERAS Y PUNTOS DE REFERENCIA")

    print(f"\n  {'ID':<5}{'Nombre':<30}{'Lat':>8}  {'Lon':>10}  {'Tipo':<14}{'Recargas'}")
    print("  " + "-" * 65)
    for e in ELECTROLINERAS:
        print(f"  {e['id']:<5}{e['nombre']:<30}{e['lat']:>8.4f}  {e['lon']:>10.4f}  {e['tipo']:<14}{e['recargas']}")

    print(f"\n  {'ID':<5}{'Nombre':<40}{'Lat':>8}  {'Lon'}")
    print("  " + "-" * 65)
    for p in PUNTOS_REF:
        print(f"  {p['id']:<5}{p['nombre']:<40}{p['lat']:>8.4f}  {p['lon']:>10.4f}")

    print(f"\n  {'Vehículo':<30}{'Gama':<8}{'kWh':>6}  {'Rango (km)':>12}")
    print("  " + "-" * 60)
    for v in VEHICULOS:
        print(f"  {v['nombre']:<30}{v['gama']:<8}{v['bateria_kwh']:>6.1f}  {v['rango_km']:>12}")

    input("\n  Enter para continuar...")


# ══════════════════════════════════════════════════════════════════════════════
# MÓDULO 3 — SIMULACIÓN DE RECORRIDOS
# ══════════════════════════════════════════════════════════════════════════════

def electrolinera_mas_cercana(lat, lon):
    """
    Encuentra la electrolinera más cercana a una coordenada dada.

    Recorre toda la lista ELECTROLINERAS calculando la distancia Haversine
    a cada una y retorna la de menor distancia.

    Args:
        lat (float): Latitud del punto de referencia.
        lon (float): Longitud del punto de referencia.

    Returns:
        tuple: (dict electrolinera más cercana, float distancia en km).
    """
    min_d = float("inf")
    resultado = None
    for e in ELECTROLINERAS:
        d = haversine(lat, lon, e["lat"], e["lon"])
        if d < min_d:
            min_d = d
            resultado = e
    return resultado, min_d


def mod_simular_recorridos():
    """
    Módulo 3 del menú: simula N viajes aleatorios entre puntos de referencia
    con un vehículo eléctrico seleccionado por el usuario.

    Lógica de recarga:
      - Recarga preventiva: si la batería actual no alcanza para el trayecto,
        recarga al 100 % en la electrolinera más cercana al ORIGEN antes de salir.
      - Recarga por nivel crítico: si al llegar la batería cae a ≤ 20 %,
        recarga al 100 % en la electrolinera más cercana al DESTINO.

    La distancia de cada viaje se calcula como Haversine × 1.35 para aproximar
    la distancia real por carretera (factor de tortuosidad).

    Cada recarga queda registrada en la lista global registro_recargas con:
    número de recorrido, vehículo, origen, destino, km, electrolinera usada,
    distancia a la electrolinera, batería antes de recargar y motivo de recarga.

    Requiere que el grafo G esté cargado (módulo 1).
    """
    global registro_recargas
    separador("MÓDULO 3 — SIMULACIÓN DE RECORRIDOS")

    if G is None:
        print("  ⚠ Primero cargue el grafo (opción 1).")
        input("  Enter..."); return

    print("\n  Vehículos disponibles:")
    for v in VEHICULOS:
        print(f"  [{v['id']}] {v['nombre']} ({v['gama']} gama) | Batería: {v['bateria_actual_pct']:.0f}%")

    id_v = validar_entero("\n  Seleccione vehículo (0 o 1): ", 0, 1)
    n    = validar_entero("  ¿Cuántos recorridos simular? (1-30): ", 1, 30)

    vehiculo = VEHICULOS[id_v]
    vehiculo["bateria_actual_pct"] = 100.0  # Reinicia batería al inicio de la sesión

    # Porcentaje de batería consumido por kilómetro recorrido
    consumo_pct_km = 100.0 / vehiculo["rango_km"]

    print(f"\n  Simulando {n} recorridos con {vehiculo['nombre']}...\n")
    print(f"  {'#':<4} {'Origen':<22} {'Destino':<22} {'km':>6}  {'Bat%':>6}")
    print("  " + "-" * 65)

    for i in range(n):
        origen  = random.choice(PUNTOS_REF)
        destino = random.choice([p for p in PUNTOS_REF if p["id"] != origen["id"]])

        # Factor ×1.35: aproxima la distancia real por carretera desde la geodésica
        dist_km = haversine(origen["lat"], origen["lon"],
                            destino["lat"], destino["lon"]) * 1.35
        consumo = consumo_pct_km * dist_km

        # ── Recarga preventiva ─────────────────────────────────────────────
        if consumo > vehiculo["bateria_actual_pct"]:
            e_previa, dist_previa = electrolinera_mas_cercana(origen["lat"], origen["lon"])
            e_previa["recargas"] += 1
            vehiculo["recargas_totales"] += 1
            vehiculo["bateria_actual_pct"] = 100.0
            registro_recargas.append({
                "recorrido":          i + 1,
                "vehiculo":           vehiculo["nombre"],
                "origen":             origen["nombre"],
                "destino":            destino["nombre"],
                "km":                 round(dist_km, 2),
                "electrolinera":      e_previa["nombre"],
                "dist_electrolinera": round(dist_previa, 2),
                "bateria_antes_pct":  round(vehiculo["bateria_actual_pct"], 1),
                "motivo":             "recarga_preventiva",
            })
            print(f"  {i+1:<4} {'[RECARGA PREVENTIVA]':<22} {origen['nombre'][:22]:<22}")
            print(f"       ⚡ Batería insuficiente → recarga en {e_previa['nombre']} antes de partir")

        # ── Ejecutar recorrido ─────────────────────────────────────────────
        vehiculo["bateria_actual_pct"] -= consumo
        vehiculo["km_recorridos"] += dist_km
        bat = vehiculo["bateria_actual_pct"]

        print(f"  {i+1:<4} {origen['nombre'][:22]:<22} {destino['nombre'][:22]:<22} {dist_km:>6.1f}  {bat:>6.1f}")

        # ── Recarga por nivel crítico (≤ 20 %) ────────────────────────────
        if bat <= 20.0:
            e_cercana, dist_e = electrolinera_mas_cercana(destino["lat"], destino["lon"])
            e_cercana["recargas"] += 1
            vehiculo["recargas_totales"] += 1
            vehiculo["bateria_actual_pct"] = 100.0
            registro_recargas.append({
                "recorrido":          i + 1,
                "vehiculo":           vehiculo["nombre"],
                "origen":             origen["nombre"],
                "destino":            destino["nombre"],
                "km":                 round(dist_km, 2),
                "electrolinera":      e_cercana["nombre"],
                "dist_electrolinera": round(dist_e, 2),
                "bateria_antes_pct":  round(bat, 1),
                "motivo":             "nivel_critico",
            })
            print(f"       ⚡ RECARGA → {e_cercana['nombre']} ({dist_e:.2f} km) | Bat → 100%")

    print(f"\n  ✓ Simulación completa. Recargas totales: {vehiculo['recargas_totales']}")
    input("\n  Enter para continuar...")


# ══════════════════════════════════════════════════════════════════════════════
# MÓDULO 4 — RUTAS MÁS CORTAS
# ══════════════════════════════════════════════════════════════════════════════

def dijkstra_heapq(adj, inicio, fin, nodos_ids):
    """
    Algoritmo de Dijkstra con cola de prioridad (min-heap).

    Encuentra el camino de menor costo entre dos nodos en un grafo
    no dirigido con pesos no negativos.

    Complejidad: O((V + E) log V), óptimo para grafos dispersos.

    Args:
        adj       (dict): Lista de adyacencia {nodo: [(vecino, peso), ...]}.
        inicio    (int) : ID del nodo de origen.
        fin       (int) : ID del nodo de destino.
        nodos_ids (list): Lista de todos los IDs de nodos del grafo.

    Returns:
        tuple: (list camino de IDs desde inicio hasta fin,
                float distancia total en km).
               Retorna ([], inf) si no existe ruta.
    """
    dist = {n: float("inf") for n in nodos_ids}
    prev = {n: None         for n in nodos_ids}
    dist[inicio] = 0.0
    heap = [(0.0, inicio)]

    while heap:
        d_actual, actual = heapq.heappop(heap)
        if d_actual > dist[actual]:
            continue  # Entrada obsoleta en el heap
        if actual == fin:
            break
        for vecino, peso in adj.get(actual, []):
            nueva = dist[actual] + peso
            if nueva < dist[vecino]:
                dist[vecino] = nueva
                prev[vecino] = actual
                heapq.heappush(heap, (nueva, vecino))

    if dist[fin] == float("inf"):
        return [], float("inf")

    # Reconstruir el camino desde el destino hacia el origen
    camino, cur = [], fin
    while cur is not None:
        camino.insert(0, cur)
        cur = prev[cur]
    return camino, round(dist[fin], 4)


def floyd_warshall(adj, nodos_ids):
    """
    Algoritmo de Floyd-Warshall para calcular distancias mínimas entre todos los pares.

    Construye una matriz V×V con las distancias más cortas entre cada par de nodos.
    Bloqueado si el grafo supera los 100 nodos para evitar consumo excesivo de RAM:
    con el grafo OSMnx de Bucaramanga (~50 000 nodos) la matriz superaría los 20 GB.

    Complejidad: O(V³) en tiempo, O(V²) en espacio.

    Args:
        adj       (dict): Lista de adyacencia {nodo: [(vecino, peso), ...]}.
        nodos_ids (list): Lista de todos los IDs de nodos.

    Returns:
        tuple: (list[list] matriz de distancias, dict índice nodo→posición en matriz).
               Retorna (None, None) si el grafo tiene más de 100 nodos.
    """
    if len(nodos_ids) > 100:
        return None, None

    idx  = {n: i for i, n in enumerate(nodos_ids)}
    size = len(nodos_ids)
    INF  = float("inf")

    # Inicializar matriz con infinito; diagonal en 0
    dist = [[INF] * size for _ in range(size)]
    for i in range(size):
        dist[i][i] = 0.0
    for u in nodos_ids:
        for v, w in adj.get(u, []):
            dist[idx[u]][idx[v]] = w
            dist[idx[v]][idx[u]] = w

    # Relajación triple: nodo intermedio k
    for k in range(size):
        for i in range(size):
            for j in range(size):
                candidato = dist[i][k] + dist[k][j]
                if candidato < dist[i][j]:
                    dist[i][j] = candidato

    return dist, idx


def construir_adj():
    """
    Construye la lista de adyacencia del grafo G activo.

    Solo incluye aristas cuyos dos extremos sean nodos del sistema
    (electrolineras o puntos de referencia).

    Returns:
        tuple: (dict adj {nodo: [(vecino, peso), ...]}, list todos_ids).
    """
    todos_ids = [n["id"] for n in ELECTROLINERAS + PUNTOS_REF]
    adj = {n: [] for n in todos_ids}
    for u, v, data in G.edges(data=True):
        if u in adj and v in adj:
            w = data.get("weight", 1)
            adj[u].append((v, w))
            adj[v].append((u, w))
    return adj, todos_ids


def mod_rutas_cortas():
    """
    Módulo 4 del menú: calcula la ruta más corta entre dos nodos usando
    Dijkstra y Floyd-Warshall, muestra los resultados y los compara.

    Floyd-Warshall se bloquea automáticamente si el grafo tiene > 100 nodos.
    Requiere que el grafo G esté cargado (módulo 1).
    """
    separador("MÓDULO 4 — RUTAS MÁS CORTAS (DIJKSTRA + FLOYD-WARSHALL)")

    if G is None:
        print("  ⚠ Primero cargue el grafo (opción 1).")
        input("  Enter..."); return

    todos = ELECTROLINERAS + PUNTOS_REF
    for n in todos:
        print(f"  [{n['id']:>2}] {n['nombre']}")

    id_o = validar_entero("\n  ID origen (0-19):  ", 0, 19)
    id_d = validar_entero("  ID destino (0-19): ", 0, 19)
    if id_o == id_d:
        print("  ⚠ Origen y destino deben ser distintos.")
        input("  Enter..."); return

    adj, todos_ids = construir_adj()

    # ── Dijkstra ───────────────────────────────────────────────────────────
    t0 = time.time()
    camino_d, dist_d = dijkstra_heapq(adj, id_o, id_d, todos_ids)
    t_dijkstra = time.time() - t0

    print(f"\n  ── DIJKSTRA (O((V+E) log V)) ──────────────────────────")
    if not camino_d:
        print("  ✗ Sin ruta entre los nodos seleccionados.")
    else:
        print(f"  Distancia: {dist_d:.2f} km  |  Tiempo: {t_dijkstra*1000:.2f} ms")
        print("  Ruta: " + " → ".join(nombre_por_id(nid) for nid in camino_d))

    # ── Floyd-Warshall ─────────────────────────────────────────────────────
    print(f"\n  ── FLOYD-WARSHALL (O(V³)) ─────────────────────────────")
    n_nodos = G.number_of_nodes()
    if n_nodos > 100:
        print(f"  ✗ Floyd-Warshall bloqueado: el grafo actual tiene {n_nodos} nodos.")
        print(f"  Una matriz de {n_nodos}x{n_nodos} flotantes requeriria ~{n_nodos**2*8//1_000_000} MB de RAM.")
        print("  Disponible solo con el grafo sintetico (18 nodos).")
    else:
        t0 = time.time()
        fw_dist, fw_idx = floyd_warshall(adj, todos_ids)
        t_fw = time.time() - t0
        fw_val = fw_dist[fw_idx[id_o]][fw_idx[id_d]]
        print(f"  Distancia: {fw_val:.2f} km  |  Tiempo total matriz: {t_fw*1000:.2f} ms")

        print(f"\n  ── COMPARACIÓN ───────────────────────────────────────")
        dif = abs(dist_d - fw_val)
        print(f"  Dijkstra: {dist_d:.2f} km | Floyd-Warshall: {fw_val:.2f} km | Δ: {dif:.4f} km")

    input("\n  Enter para continuar...")


# ══════════════════════════════════════════════════════════════════════════════
# MÓDULO 5 — QUICKSORT
# ══════════════════════════════════════════════════════════════════════════════

def _particionar(lista, inicio, fin):
    """
    Particiona una sublista de electrolineras para QuickSort.

    Usa el último elemento como pivote y reorganiza los elementos de forma que
    los que tienen más recargas queden a la izquierda del pivote (orden descendente).

    Args:
        lista  (list): Lista de dicts de electrolineras.
        inicio (int) : Índice izquierdo de la sublista.
        fin    (int) : Índice derecho (pivote).

    Returns:
        int: Índice final del pivote tras la partición.
    """
    pivote = lista[fin]["recargas"]
    i = inicio - 1
    for j in range(inicio, fin):
        if lista[j]["recargas"] >= pivote:
            i += 1
            lista[i], lista[j] = lista[j], lista[i]
    lista[i + 1], lista[fin] = lista[fin], lista[i + 1]
    return i + 1


def quicksort(lista, inicio, fin):
    """
    Ordena una lista de electrolineras por número de recargas (descendente)
    usando el algoritmo QuickSort in-place.

    Complejidad promedio: O(n log n). Peor caso: O(n²).

    Args:
        lista  (list): Lista de dicts de electrolineras.
        inicio (int) : Índice izquierdo de la sublista a ordenar.
        fin    (int) : Índice derecho de la sublista a ordenar.
    """
    if inicio < fin:
        p = _particionar(lista, inicio, fin)
        quicksort(lista, inicio, p - 1)
        quicksort(lista, p + 1, fin)


def mod_quicksort():
    """
    Módulo 5 del menú: ordena y muestra las electrolineras por número de recargas
    usando QuickSort, con una barra gráfica de uso por consola.
    """
    separador("MÓDULO 5 — QUICKSORT: ELECTROLINERAS POR USO")
    lista = [e.copy() for e in ELECTROLINERAS]
    quicksort(lista, 0, len(lista) - 1)

    print(f"\n  {'Pos':<5}{'Electrolinera':<30}{'Recargas':>10}  Grafico")
    print("  " + "-" * 60)
    for i, e in enumerate(lista):
        barra = "█" * min(e["recargas"], 30)
        print(f"  {i+1:<5}{e['nombre']:<30}{e['recargas']:>10}  {barra}")

    input("\n  Enter para continuar...")


# ══════════════════════════════════════════════════════════════════════════════
# MÓDULO 6 — BÚSQUEDA BINARIA
# ══════════════════════════════════════════════════════════════════════════════

def mod_busqueda_binaria():
    """
    Módulo 6 del menú: busca una electrolinera por nombre usando Búsqueda Binaria.

    La lista se ordena alfabéticamente antes de la búsqueda.
    Si no hay coincidencia exacta, realiza una búsqueda parcial lineal como fallback.
    """
    separador("MÓDULO 6 — BÚSQUEDA BINARIA DE ELECTROLINERA")

    termino = input("  Nombre a buscar (parcial o completo): ").strip().lower()
    if not termino:
        print("  ⚠ Entrada vacía."); input("  Enter..."); return

    lista = sorted(ELECTROLINERAS, key=lambda x: x["nombre"].lower())
    bajo, alto, encontrado = 0, len(lista) - 1, None

    # Búsqueda binaria sobre nombres ordenados alfabéticamente
    while bajo <= alto:
        medio    = (bajo + alto) // 2
        nombre_m = lista[medio]["nombre"].lower()
        if nombre_m == termino:
            encontrado = lista[medio]
            break
        elif nombre_m < termino:
            bajo = medio + 1
        else:
            alto = medio - 1

    if encontrado:
        print(f"\n  ✓ Coincidencia exacta: {encontrado['nombre']}")
        print(f"    Lat: {encontrado['lat']} | Lon: {encontrado['lon']}")
        print(f"    Tipo: {encontrado['tipo']} | Recargas: {encontrado['recargas']}")
    else:
        # Fallback: búsqueda parcial lineal
        print("  ✗ Sin coincidencia exacta. Buscando coincidencias parciales...")
        parciales = [e for e in ELECTROLINERAS if termino in e["nombre"].lower()]
        if parciales:
            for r in parciales:
                print(f"    → {r['nombre']} | Recargas: {r['recargas']}")
        else:
            print("  ✗ Sin resultados.")

    input("\n  Enter para continuar...")


# ══════════════════════════════════════════════════════════════════════════════
# MÓDULO 7 — BFS Y DFS
# ══════════════════════════════════════════════════════════════════════════════

def mod_bfs_dfs():
    """
    Módulo 7 del menú: realiza recorridos BFS y DFS sobre el grafo desde un nodo raíz.

    BFS (Breadth-First Search):
      Usa collections.deque con popleft() → extracción O(1).
      Explora por capas (vecinos más cercanos primero).

    DFS (Depth-First Search):
      Iterativo con pila explícita (list.pop() → O(1)).
      Explora en profundidad antes de retroceder.

    Ambos tienen complejidad O(V + E).
    Los vecinos se ordenan para garantizar determinismo en el recorrido.

    Requiere que el grafo G esté cargado (módulo 1).
    """
    separador("MÓDULO 7 — RECORRIDOS BFS Y DFS")

    if G is None:
        print("  ⚠ Primero cargue el grafo (opción 1).")
        input("  Enter..."); return

    todos     = ELECTROLINERAS + PUNTOS_REF
    todos_ids = [n["id"] for n in todos]

    for n in todos:
        print(f"  [{n['id']:>2}] {n['nombre']}")

    id_inicio = validar_entero("\n  ID del nodo de inicio (0-19): ", 0, 19)

    # Lista de adyacencia sin pesos (solo conectividad)
    adj = {n: [] for n in todos_ids}
    for u, v in G.edges():
        if u in adj and v in adj:
            adj[u].append(v)
            adj[v].append(u)

    # ── BFS ────────────────────────────────────────────────────────────────
    cola_bfs      = collections.deque([id_inicio])
    vis_bfs       = {id_inicio}
    visitados_bfs = []
    while cola_bfs:
        actual = cola_bfs.popleft()
        visitados_bfs.append(actual)
        for vecino in sorted(adj.get(actual, [])):
            if vecino not in vis_bfs:
                vis_bfs.add(vecino)
                cola_bfs.append(vecino)

    # ── DFS iterativo ──────────────────────────────────────────────────────
    visitados_dfs, pila, vis_dfs = [], [id_inicio], set()
    while pila:
        actual = pila.pop()
        if actual not in vis_dfs:
            vis_dfs.add(actual)
            visitados_dfs.append(actual)
            for vecino in sorted(adj.get(actual, []), reverse=True):
                if vecino not in vis_dfs:
                    pila.append(vecino)

    print(f"\n  Inicio: {nombre_por_id(id_inicio)}\n")
    print("  BFS: " + " → ".join(nombre_por_id(x) for x in visitados_bfs))
    print("\n  DFS: " + " → ".join(nombre_por_id(x) for x in visitados_dfs))

    input("\n  Enter para continuar...")


# ══════════════════════════════════════════════════════════════════════════════
# MÓDULOS 8 Y 9 — MACHINE LEARNING (RANDOM FOREST)
# ══════════════════════════════════════════════════════════════════════════════

def _generar_dataset_ml():
    """
    Genera un dataset sintético para entrenar el modelo de clasificación de zonas.

    Crea 1 200 puntos aleatorios dentro del bounding box geográfico del sistema.
    La etiqueta de cada punto es el índice de la electrolinera más cercana
    (frontera de Voronoi por distancia Haversine), simulando la región de
    influencia de cada estación de carga.

    El modelo aprende a clasificar regiones del espacio sin acceso a distancias:
    generaliza la geometría sin memorizarla.

    Returns:
        list: Lista de [lat, lon, etiqueta] para cada muestra generada.
    """
    todos = ELECTROLINERAS + PUNTOS_REF

    lat_min = min(n["lat"] for n in todos) - 0.02
    lat_max = max(n["lat"] for n in todos) + 0.02
    lon_min = min(n["lon"] for n in todos) - 0.02
    lon_max = max(n["lon"] for n in todos) + 0.02

    datos = []
    for _ in range(1200):
        lat = random.uniform(lat_min, lat_max)
        lon = random.uniform(lon_min, lon_max)
        dists    = [haversine(lat, lon, e["lat"], e["lon"]) for e in ELECTROLINERAS]
        etiqueta = dists.index(min(dists))  # Clase = electrolinera más cercana
        datos.append([lat, lon, etiqueta])
    return datos


def mod_entrenar_ml():
    """
    Módulo 8 del menú: entrena un clasificador Random Forest sobre las zonas de Voronoi.

    Pipeline:
      1. Genera dataset con _generar_dataset_ml() (1200 muestras, features: lat/lon).
      2. Divide en 80 % entrenamiento / 20 % prueba.
      3. Normaliza las features con StandardScaler.
      4. Entrena un Random Forest de 100 árboles.
      5. Muestra accuracy y la importancia de cada feature.

    El modelo entrenado y el scaler quedan guardados en las variables globales
    modelo_ml y scaler_ml para ser usados en el módulo 9.
    """
    global modelo_ml, scaler_ml
    separador("MÓDULO 8 — ENTRENAMIENTO MODELO ML (RANDOM FOREST)")

    print("  Generando dataset geografico de entrenamiento...")
    dataset = _generar_dataset_ml()

    X = np.array([f[:2] for f in dataset])
    y = np.array([f[2]  for f in dataset], dtype=int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler_ml = StandardScaler()
    X_tr_sc   = scaler_ml.fit_transform(X_train)
    X_te_sc   = scaler_ml.transform(X_test)

    modelo_ml = RandomForestClassifier(n_estimators=100, random_state=42)
    modelo_ml.fit(X_tr_sc, y_train)

    acc = accuracy_score(y_test, modelo_ml.predict(X_te_sc))

    print(f"\n  Dataset: {len(dataset)} muestras | Train: {len(X_train)} | Test: {len(X_test)}")
    print(f"  Features de entrada: latitud, longitud (solo coordenadas)")
    print(f"  Algoritmo: Random Forest (100 arboles de decision)")
    print(f"  Accuracy en prueba: {acc * 100:.2f}%")
    print(f"\n  El modelo aprendio las regiones de Voronoi del espacio geografico:")
    print(f"  dado un punto (lat, lon), predice a que electrolinera pertenece")
    print(f"  su zona sin calcular distancias en tiempo de inferencia.")

    print("\n  Importancia de caracteristicas:")
    importancias = list(enumerate(modelo_ml.feature_importances_))
    importancias.sort(key=lambda x: x[1], reverse=True)
    nombres_feat = ["latitud", "longitud"]
    for idx, imp in importancias:
        barra = "█" * int(imp * 60)
        print(f"    {nombres_feat[idx]:<12} {imp:.4f} {barra}")

    input("\n  Enter para continuar...")


def mod_predecir_ml():
    """
    Módulo 9 del menú: predice la zona de carga de un punto de referencia
    usando el modelo Random Forest entrenado en el módulo 8.

    Muestra:
      - La electrolinera predicha por el modelo y su confianza.
      - La electrolinera más cercana por Haversine (referencia geográfica real).
      - Distribución de probabilidades del modelo para todas las electrolineras.
      - Indicación de si el modelo coincide con la referencia o no (error de frontera).

    Requiere que el modelo esté entrenado (módulo 8).
    """
    separador("MÓDULO 9 — PREDICCIÓN DE ZONA CON ML")

    if modelo_ml is None:
        print("  ⚠ Primero entrene el modelo (opción 8).")
        input("  Enter..."); return

    for p in PUNTOS_REF:
        print(f"  [{p['id']}] {p['nombre']}")

    id_p  = validar_entero("\n  Seleccione punto de partida (10-19): ", 10, 19)
    punto = next(p for p in PUNTOS_REF if p["id"] == id_p)

    X_new = np.array([[punto["lat"], punto["lon"]]])
    X_sc  = scaler_ml.transform(X_new)
    pred  = modelo_ml.predict(X_sc)[0]
    probs = modelo_ml.predict_proba(X_sc)[0]

    e_pred = ELECTROLINERAS[pred]
    dists  = [haversine(punto["lat"], punto["lon"], e["lat"], e["lon"]) for e in ELECTROLINERAS]
    e_real = ELECTROLINERAS[dists.index(min(dists))]

    print(f"\n  Origen: {punto['nombre']}")
    print(f"  Coordenadas: ({punto['lat']:.4f}, {punto['lon']:.4f})")

    print(f"\n  ── PREDICCION ML (Random Forest) ──────────────────────")
    print(f"  Electrolinera de zona predicha : {e_pred['nombre']}")
    print(f"  Distancia real a esa estacion  : {dists[pred]:.2f} km")
    print(f"  Confianza del modelo           : {probs[pred] * 100:.1f}%")

    print(f"\n  ── REFERENCIA GEOGRAFICA (Haversine directa) ───────────")
    print(f"  Electrolinera mas cercana por distancia: {e_real['nombre']} ({min(dists):.2f} km)")

    if pred == dists.index(min(dists)):
        print("  ✓ El modelo coincide con la referencia geografica.")
    else:
        print("  ≈ El modelo asigna una zona diferente (error de frontera de Voronoi).")

    print("\n  Distribucion de probabilidades por electrolinera:")
    for i, e in enumerate(ELECTROLINERAS):
        barra = "█" * int(probs[i] * 40)
        print(f"    {e['nombre']:<28} {probs[i]*100:5.1f}%  {barra}")

    input("\n  Enter para continuar...")


# ══════════════════════════════════════════════════════════════════════════════
# MÓDULO 10 — K-MEANS
# ══════════════════════════════════════════════════════════════════════════════

def mod_kmeans():
    """
    Módulo 10 del menú: sugiere nuevas ubicaciones de electrolineras usando K-Means.

    Construye el dataset combinando:
      - Los puntos de referencia (sedes universitarias) con peso 1.
      - Las electrolineras existentes, repetidas según su uso (recargas × 3),
        para que las zonas de alta demanda influyan más en los centroides.

    El algoritmo converge hacia los centroides de mayor densidad de uso/presencia,
    sugiriendo coordenadas óptimas para nuevas estaciones.

    Muestra los centroides resultantes y la zona de referencia más cercana a cada uno.
    """
    separador("MÓDULO 10 — K-MEANS: NUEVAS UBICACIONES SUGERIDAS")

    n_nuevas = validar_entero("  ¿Cuántas nuevas electrolineras sugerir? (1-5): ", 1, 5)

    # Dataset ponderado: más repeticiones = mayor peso en el centroide
    puntos = []
    for p in PUNTOS_REF:
        puntos.append([p["lat"], p["lon"]])
    for e in ELECTROLINERAS:
        repeticiones = max(1, e["recargas"] * 3)
        for _ in range(repeticiones):
            puntos.append([e["lat"], e["lon"]])

    X  = np.array(puntos)
    km = KMeans(n_clusters=n_nuevas, random_state=42, n_init=10)
    km.fit(X)

    print(f"\n  ✓ K-Means convergió en {km.n_iter_} iteraciones\n")
    print(f"  {'#':<4} {'Latitud':>10}  {'Longitud':>11}  {'Zona mas cercana'}")
    print("  " + "-" * 60)

    for i, centro in enumerate(km.cluster_centers_):
        dists_ref = [haversine(centro[0], centro[1], p["lat"], p["lon"]) for p in PUNTOS_REF]
        ref       = PUNTOS_REF[dists_ref.index(min(dists_ref))]
        print(f"  {i+1:<4} {centro[0]:>10.5f}  {centro[1]:>11.5f}  {ref['nombre']} ({min(dists_ref):.2f} km)")

    input("\n  Enter para continuar...")


# ══════════════════════════════════════════════════════════════════════════════
# MÓDULO 11 — MAPA INTERACTIVO
# ══════════════════════════════════════════════════════════════════════════════

def mod_visualizar_mapa():
    """
    Módulo 11 del menú: genera un mapa interactivo HTML usando Folium.

    El mapa muestra:
      - Marcadores rojos (⚡) para cada electrolinera con popup de recargas y tipo.
      - Marcadores azules (🎓) para cada punto de referencia.
      - Aristas del grafo G dibujadas como líneas grises semitransparentes.
      - Leyenda en la esquina inferior izquierda.

    El archivo resultante se guarda en BASE_DIR/mapa_electrolineras.html
    y puede abrirse en cualquier navegador web.
    """
    separador("MÓDULO 11 — MAPA INTERACTIVO (FOLIUM)")

    mapa = folium.Map(location=[7.09, -73.11], zoom_start=12,
                      tiles="OpenStreetMap")

    # Marcadores de electrolineras
    for e in ELECTROLINERAS:
        folium.Marker(
            location=[e["lat"], e["lon"]],
            popup=f"⚡ {e['nombre']}<br>Recargas: {e['recargas']}<br>Tipo: {e['tipo']}",
            tooltip=e["nombre"],
            icon=folium.Icon(color="red", icon="bolt", prefix="fa")
        ).add_to(mapa)

    # Marcadores de puntos de referencia
    for p in PUNTOS_REF:
        folium.Marker(
            location=[p["lat"], p["lon"]],
            popup=f"🎓 {p['nombre']}",
            tooltip=p["nombre"],
            icon=folium.Icon(color="blue", icon="university", prefix="fa")
        ).add_to(mapa)

    # Aristas del grafo como líneas grises
    if G:
        todos = {n["id"]: n for n in ELECTROLINERAS + PUNTOS_REF}
        for u, v in G.edges():
            if u in todos and v in todos:
                folium.PolyLine(
                    [[todos[u]["lat"], todos[u]["lon"]],
                     [todos[v]["lat"], todos[v]["lon"]]],
                    color="gray", weight=1, opacity=0.5
                ).add_to(mapa)

    # Leyenda HTML fija en la esquina inferior izquierda
    leyenda = """
    <div style="position:fixed;bottom:30px;left:30px;z-index:1000;
                background:white;padding:10px;border-radius:8px;
                border:2px solid gray;font-size:13px">
      <b>Leyenda</b><br>
      Electrolinera (rojo)<br>
      Punto de referencia (azul)
    </div>"""
    mapa.get_root().html.add_child(folium.Element(leyenda))

    archivo_html = os.path.join(BASE_DIR, "mapa_electrolineras.html")
    mapa.save(archivo_html)
    print(f"\n  ✓ Mapa guardado como '{archivo_html}'")
    print("  Ábrelo en cualquier navegador para verlo interactivo.")

    input("\n  Enter para continuar...")


# ══════════════════════════════════════════════════════════════════════════════
# MÓDULO 12 — GENERACIÓN DE REPORTES
# ══════════════════════════════════════════════════════════════════════════════

def mod_generar_archivos():
    """
    Módulo 12 del menú: exporta el estado actual del sistema a múltiples formatos.

    Archivos generados en BASE_DIR:
      - electrolineras_bucaramanga.json : Electrolineras, puntos de referencia y vehículos.
      - registro_recargas.csv           : Historial detallado de cada recarga simulada.
      - reporte_general.txt             : Reporte de texto con ranking de uso y resumen.
      - estadisticas.xlsx               : Tres hojas (Electrolineras, Vehiculos, Recargas).
                                          Requiere openpyxl instalado.

    Al final verifica la integridad del JSON releyéndolo desde disco.
    """
    separador("MÓDULO 12 — GENERACIÓN DE REPORTES EN ARCHIVOS")

    # ── JSON ────────────────────────────────────────────────────────────────
    datos_json = {
        "electrolineras":    ELECTROLINERAS,
        "puntos_referencia": PUNTOS_REF,
        "vehiculos":         [{k: v for k, v in ve.items()} for ve in VEHICULOS],
    }
    ruta_json = os.path.join(BASE_DIR, "electrolineras_bucaramanga.json")
    with open(ruta_json, "w", encoding="utf-8") as f:
        json.dump(datos_json, f, ensure_ascii=False, indent=2)
    print("  ✓ electrolineras_bucaramanga.json")

    # ── CSV ─────────────────────────────────────────────────────────────────
    if registro_recargas:
        ruta_csv = os.path.join(BASE_DIR, "registro_recargas.csv")
        with open(ruta_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=registro_recargas[0].keys())
            writer.writeheader()
            writer.writerows(registro_recargas)
        print("  ✓ registro_recargas.csv")
    else:
        print("  ! registro_recargas.csv — sin datos (simule recorridos primero)")

    # ── TXT ─────────────────────────────────────────────────────────────────
    lineas = [
        "=" * 60,
        "REPORTE SISTEMA ELECTROLINERAS - BUCARAMANGA 2026",
        "Proyecto Integrador | Facultad Ing. Ciencia de Datos - UIS",
        "=" * 60,
        f"Total recargas registradas: {len(registro_recargas)}",
        "",
        "ELECTROLINERAS (ordenadas por uso):",
    ]
    lista_txt = sorted(ELECTROLINERAS, key=lambda x: x["recargas"], reverse=True)
    for e in lista_txt:
        lineas.append(f"  {e['nombre']:<30} {e['recargas']} recargas")
    lineas += ["", "VEHICULOS:"]
    for v in VEHICULOS:
        lineas.append(
            f"  {v['nombre']:<30} {v['recargas_totales']} recargas "
            f"| {v['km_recorridos']:.1f} km recorridos"
        )
    ruta_txt = os.path.join(BASE_DIR, "reporte_general.txt")
    with open(ruta_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(lineas))
    print("  ✓ reporte_general.txt")

    # ── XLSX ────────────────────────────────────────────────────────────────
    if XLSX_OK:
        df_e = pd.DataFrame(ELECTROLINERAS)
        df_v = pd.DataFrame([{k: v for k, v in ve.items()} for ve in VEHICULOS])
        df_r = pd.DataFrame(registro_recargas) if registro_recargas else pd.DataFrame()
        ruta_xlsx = os.path.join(BASE_DIR, "estadisticas.xlsx")
        with pd.ExcelWriter(ruta_xlsx, engine="openpyxl") as writer:
            df_e.to_excel(writer, sheet_name="Electrolineras", index=False)
            df_v.to_excel(writer, sheet_name="Vehiculos",      index=False)
            if not df_r.empty:
                df_r.to_excel(writer, sheet_name="Recargas", index=False)
        print("  ✓ estadisticas.xlsx")
    else:
        print("  ! estadisticas.xlsx — openpyxl no instalado")

    # ── Verificación de integridad ───────────────────────────────────────────
    with open(ruta_json, "r", encoding="utf-8") as f:
        datos_leidos = json.load(f)
    print(f"\n  Verificacion lectura JSON: {len(datos_leidos['electrolineras'])} electrolineras ✓")

    input("\n  Enter para continuar...")


# ══════════════════════════════════════════════════════════════════════════════
# MÓDULO 13 — ESTADÍSTICAS GENERALES
# ══════════════════════════════════════════════════════════════════════════════

def mod_estadisticas():
    """
    Módulo 13 del menú: muestra un resumen estadístico del sistema.

    Incluye:
      - Total de recargas simuladas y km recorridos por todos los vehículos.
      - Métricas del grafo: nodos, aristas, conectividad, grado promedio.
      - Resumen por vehículo: recargas, km y nivel de batería actual.
      - Electrolinera más y menos utilizada (si hay datos de simulación).
      - Estado del modelo ML (entrenado o no).
    """
    separador("MÓDULO 13 — ESTADÍSTICAS GENERALES")

    total_r  = sum(e["recargas"] for e in ELECTROLINERAS)
    total_km = sum(v["km_recorridos"] for v in VEHICULOS)

    print(f"\n  Total recargas simuladas : {total_r}")
    print(f"  Total registros guardados: {len(registro_recargas)}")
    print(f"  Total km recorridos      : {total_km:.1f} km")

    if G:
        print(f"\n  Grafo: {G.number_of_nodes()} nodos | {G.number_of_edges()} aristas")
        print(f"  Grafo conexo: {'Si' if nx.is_connected(G) else 'No'}")
        grados = [d for _, d in G.degree()]
        print(f"  Grado promedio nodos: {sum(grados)/len(grados):.2f}")

    print(f"\n  RESUMEN POR VEHICULO:")
    for v in VEHICULOS:
        print(f"  [{v['gama']}] {v['nombre']}")
        print(f"        Recargas: {v['recargas_totales']} | Km: {v['km_recorridos']:.1f} | Bat: {v['bateria_actual_pct']:.1f}%")

    if total_r > 0:
        mas   = max(ELECTROLINERAS, key=lambda e: e["recargas"])
        menos = min(ELECTROLINERAS, key=lambda e: e["recargas"])
        print(f"\n  Mas usada  : {mas['nombre']} ({mas['recargas']} recargas)")
        print(f"  Menos usada: {menos['nombre']} ({menos['recargas']} recargas)")

    print(f"\n  ML entrenado: {'Si' if modelo_ml else 'No'}")

    input("\n  Enter para continuar...")


# ══════════════════════════════════════════════════════════════════════════════
# MENÚ PRINCIPAL Y BUCLE DE CONTROL
# ══════════════════════════════════════════════════════════════════════════════

def mostrar_menu():
    """Limpia la pantalla y muestra el menú principal con todas las opciones."""
    limpiar()
    print("""
╔══════════════════════════════════════════════════════════════╗
║       SISTEMA DE ELECTROLINERAS — BUCARAMANGA 2026          ║
║       Facultad de Ingeniería en Ciencia de Datos — UIS      ║
╠══════════════════════════════════════════════════════════════╣
║  1.  Cargar grafo de Bucaramanga (OSMnx / Sintético)        ║
║  2.  Ver electrolineras, puntos de referencia y vehículos   ║
║  3.  Simular recorridos de vehículos eléctricos             ║
║  4.  Calcular rutas más cortas (Dijkstra + Floyd-Warshall)  ║
║  5.  Ordenar electrolineras por uso (QuickSort)             ║
║  6.  Buscar electrolinera (Búsqueda Binaria)                ║
║  7.  Recorridos BFS y DFS del grafo                         ║
║  8.  Entrenar modelo ML (Random Forest — zonas Voronoi)     ║
║  9.  Predecir zona de carga con ML                          ║
║  10. Sugerir nuevas ubicaciones (K-Means)                   ║
║  11. Visualizar mapa interactivo (Folium)                   ║
║  12. Generar reportes en archivos (.csv .json .txt .xlsx)   ║
║  13. Ver estadísticas generales                             ║
║  0.  Salir                                                  ║
╚══════════════════════════════════════════════════════════════╝""")


# ── Inicialización ─────────────────────────────────────────────────────────────
# Al arrancar se construye el grafo sintético automáticamente para que el sistema
# sea funcional sin necesidad de pasar por el módulo 1.
print("\n  Iniciando sistema, construyendo grafo sintetico...")
construir_grafo_sintetico()
print(f"  ✓ Grafo base listo. Nodos: {G.number_of_nodes()} | Aristas: {G.number_of_edges()}")
time.sleep(1)

# ── Bucle principal ────────────────────────────────────────────────────────────
# Centinela = 0 (opción "Salir"). El bucle valida la entrada antes de despachar.
CENTINELA = 0
opcion    = -1

while opcion != CENTINELA:
    mostrar_menu()
    entrada = input("\n  Seleccione una opcion (0-13): ").strip()

    if not entrada.isdigit():
        print("  ⚠ Solo numeros enteros del 0 al 13.")
        time.sleep(1.2)
        continue

    opcion = int(entrada)

    if   opcion == 1:  mod_cargar_grafo()
    elif opcion == 2:  mod_ver_nodos()
    elif opcion == 3:  mod_simular_recorridos()
    elif opcion == 4:  mod_rutas_cortas()
    elif opcion == 5:  mod_quicksort()
    elif opcion == 6:  mod_busqueda_binaria()
    elif opcion == 7:  mod_bfs_dfs()
    elif opcion == 8:  mod_entrenar_ml()
    elif opcion == 9:  mod_predecir_ml()
    elif opcion == 10: mod_kmeans()
    elif opcion == 11: mod_visualizar_mapa()
    elif opcion == 12: mod_generar_archivos()
    elif opcion == 13: mod_estadisticas()
    elif opcion == CENTINELA:
        print("\n  ¡Hasta luego! Sistema finalizado.\n")
    else:
        print("  ⚠ Opcion invalida. Use 0-13.")
        time.sleep(1.2)
        opcion = -1
