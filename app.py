# app.py — Sericultura (Bombyx mori & Morus sp.)
# Requisitos: streamlit, pandas, numpy, matplotlib, openpyxl (opcional: statsmodels)
# Ejecución (PowerShell):
#   cd C:\Users\lmpca\sericultura_mvp
#   .\.venv\Scripts\python.exe -m streamlit run .\app.py

import sqlite3
from pathlib import Path
import datetime as dt

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# === (opcional) statsmodels para ANOVA; si no está, la app sigue funcionando ===
try:
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    HAS_STATSMODELS = True
except Exception:
    HAS_STATSMODELS = False

DB_PATH = Path(__file__).parent / "sericultura.db"

SCHEMA_SQL = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS linea_bmori (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  codigo TEXT UNIQUE NOT NULL,
  origen TEXT,
  descripcion TEXT,
  activo INTEGER DEFAULT 1
);

CREATE TABLE IF NOT EXISTS cohorte (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  linea_id INTEGER NOT NULL,
  fecha_incubacion TEXT,
  fecha_eclosion TEXT,
  campania TEXT,
  observaciones TEXT,
  FOREIGN KEY (linea_id) REFERENCES linea_bmori(id)
);

CREATE TABLE IF NOT EXISTS cosecha_capullos (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  cohorte_id INTEGER NOT NULL,
  fecha TEXT,
  n_capullos INTEGER,
  peso_prom_capullo_lleno_g REAL,
  peso_prom_capullo_vacio_g REAL,
  observaciones TEXT,
  FOREIGN KEY (cohorte_id) REFERENCES cohorte(id)
);

-- Morus sp.
CREATE TABLE IF NOT EXISTS variedad_morus (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  codigo TEXT UNIQUE NOT NULL,
  origen TEXT,
  descripcion TEXT,
  activo INTEGER DEFAULT 1
);

CREATE TABLE IF NOT EXISTS arbol (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  variedad_id INTEGER NOT NULL,
  parcela TEXT,
  codigo_arbol TEXT,
  anio_plantacion INTEGER,
  manejo TEXT,
  observaciones TEXT,
  FOREIGN KEY (variedad_id) REFERENCES variedad_morus(id)
);

CREATE TABLE IF NOT EXISTS fenologia (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  arbol_id INTEGER NOT NULL,
  fecha_brotacion TEXT,
  fecha_caducidad_hojas TEXT,
  observaciones TEXT,
  FOREIGN KEY (arbol_id) REFERENCES arbol(id)
);

CREATE TABLE IF NOT EXISTS morfometria_hoja (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  arbol_id INTEGER NOT NULL,
  fecha TEXT NOT NULL,
  n_ramas INTEGER,
  n_hojas_por_rama INTEGER,
  largo_hoja_cm REAL,
  ancho_hoja_cm REAL,
  forma_hoja TEXT,
  peso_hoja_g REAL,
  observaciones TEXT,
  FOREIGN KEY (arbol_id) REFERENCES arbol(id)
);
"""

# ---------------- Utilidades DB ----------------
def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn

def init_db():
    with get_conn() as conn:
        conn.executescript(SCHEMA_SQL)

# ---------------- Cálculos ----------------
def shell_ratio(pl, pv):
    if pl is None or pv is None:
        return np.nan
    try:
        pl = float(pl); pv = float(pv)
        if pl <= 0:
            return np.nan
        return (pl - pv) / pl * 100.0
    except Exception:
        return np.nan

def anova_shell_por_linea(df: pd.DataFrame):
    """ANOVA OLS si hay statsmodels, si no devuelve None."""
    if not HAS_STATSMODELS:
        return None, None
    df2 = df.dropna(subset=["shell_ratio","linea"]).copy()
    if df2.empty:
        return None, None
    model = smf.ols("shell_ratio ~ C(linea)", data=df2).fit()
    anova_tbl = sm.stats.anova_lm(model, typ=2)
    return model, anova_tbl

# ---------------- Streamlit ----------------
st.set_page_config(page_title="Sericultura — Bombyx & Morus", layout="wide")
# --- Autenticación ---
import streamlit as st
import streamlit_authenticator as stauth

# Construir las credenciales desde st.secrets (TOML)
def _build_credentials_from_secrets():
    creds = {"usernames": {}}
    for uname, uinfo in st.secrets["credentials"]["usernames"].items():
        creds["usernames"][uname] = {
            "name": uinfo.get("name", uname),
            "email": uinfo.get("email", ""),
            "password": uinfo["password"],   # hash bcrypt
            "role": uinfo.get("role", "viewer")
        }
    return creds

credentials = _build_credentials_from_secrets()

authenticator = stauth.Authenticate(
    credentials,
    st.secrets["auth"]["cookie_name"],
    st.secrets["auth"]["signature_key"],
    st.secrets["auth"]["cookie_expiry_days"],
)

# Render del formulario de login (en el cuerpo principal)
name, auth_status, username = authenticator.login("Iniciar sesión", "main")

if auth_status is False:
    st.error("Usuario o contraseña inválidos.")
    st.stop()
elif auth_status is None:
    st.info("Ingrese usuario y contraseña.")
    st.stop()
else:
    # Logueado: mostrar barra lateral con logout y datos
    authenticator.logout("Cerrar sesión", "sidebar")
    st.sidebar.write(f"👤 {name}")
    # (opcional) rol del usuario
    USER_ROLE = credentials["usernames"][username].get("role", "viewer")
    st.sidebar.caption(f"Rol: {USER_ROLE}")
    # A partir de aquí, tu app existente (pestañas, formularios, etc.)

@st.cache_resource
def _bootstrap():
    init_db()
    return True

_bootstrap()

st.title("Gestión y análisis — Bombyx mori (gusanos de seda) y Morus sp. (morera)")

tabs = st.tabs([
    "Líneas y Cohortes",
    "Cosecha y Reproducción",
    "Morera",
    "Análisis",
    "Exportar"
])

# ------------- Líneas y Cohortes -------------
with tabs[0]:
    st.header("Líneas y Cohortes de Bombyx mori")

    st.subheader("Registrar línea")
    col1, col2 = st.columns(2)
    with col1:
        lin_codigo = st.text_input("Código de línea (único)", placeholder="p.ej. L-AR-01")
        lin_origen = st.text_input("Origen", placeholder="p.ej. CRILAR / INTA")
    with col2:
        lin_desc = st.text_area("Descripción", placeholder="Notas de la línea")

    if st.button("Guardar línea"):
        if not lin_codigo.strip():
            st.error("El código de línea es obligatorio.")
        else:
            with get_conn() as conn:
                conn.execute(
                    "INSERT OR IGNORE INTO linea_bmori(codigo, origen, descripcion) VALUES(?,?,?)",
                    (lin_codigo.strip(), lin_origen.strip(), lin_desc.strip())
                )
            st.success("✅ Línea guardada (o ya existía).")

    st.divider()
    st.subheader("Registrar cohorte")
    with get_conn() as conn:
        lineas = conn.execute("SELECT id, codigo FROM linea_bmori WHERE activo=1 ORDER BY codigo").fetchall()
    if not lineas:
        st.info("Primero registre al menos una línea.")
    else:
        map_linea = {c: i for i, c in lineas}
        coh_linea = st.selectbox("Línea", options=list(map_linea.keys()))
        coh_inc = st.date_input("Fecha de incubación", value=dt.date.today())
        registrar_ecl = st.checkbox("Registrar fecha de eclosión")
        coh_ecl = st.date_input("Fecha de eclosión", value=dt.date.today()) if registrar_ecl else None
        coh_camp = st.text_input("Campaña", placeholder="p.ej. 2025-A")
        coh_obs = st.text_input("Observaciones", placeholder="Opcional")

        if st.button("Guardar cohorte"):
            with get_conn() as conn:
                conn.execute(
                    "INSERT INTO cohorte(linea_id, fecha_incubacion, fecha_eclosion, campania, observaciones) VALUES(?,?,?,?,?)",
                    (map_linea[coh_linea], str(coh_inc), str(coh_ecl) if coh_ecl else None, coh_camp, coh_obs)
                )
            st.success("✅ Cohorte guardada.")

    st.divider()
    st.subheader("Listado de cohortes")
    with get_conn() as conn:
        df_coh = pd.read_sql_query(
            """SELECT c.id, l.codigo AS linea, c.fecha_incubacion, c.fecha_eclosion, c.campania, c.observaciones
               FROM cohorte c JOIN linea_bmori l ON c.linea_id=l.id
               ORDER BY c.id DESC""",
            conn
        )
    st.dataframe(df_coh if not df_coh.empty else pd.DataFrame(), use_container_width=True)

# ------------- Cosecha y Reproducción -------------
with tabs[1]:
    st.header("Cosecha (capullos) y reproducción")

    with get_conn() as conn:
        cohortes = conn.execute(
            "SELECT c.id, l.codigo || ' — ' || IFNULL(c.campania,'') AS label "
            "FROM cohorte c JOIN linea_bmori l ON c.linea_id=l.id ORDER BY c.id DESC"
        ).fetchall()
    if not cohortes:
        st.info("No hay cohortes registradas aún.")
    else:
        coh_map = {label: i for i, label in cohortes}
        sel_coh = st.selectbox("Cohorte", options=list(coh_map.keys()))
        col1, col2, col3 = st.columns(3)
        with col1:
            fecha_c = st.date_input("Fecha de cosecha", value=dt.date.today())
            ncap = st.number_input("N° capullos", min_value=0, value=0, step=1)
        with col2:
            pl = st.number_input("Peso prom. capullo lleno (g)", min_value=0.0, value=0.0, step=0.01)
            pv = st.number_input("Peso prom. capullo vacío (g)", min_value=0.0, value=0.0, step=0.01)
        with col3:
            obs_c = st.text_input("Observaciones", placeholder="Opcional")

        if st.button("Guardar cosecha"):
            with get_conn() as conn:
                conn.execute(
                    "INSERT INTO cosecha_capullos(cohorte_id, fecha, n_capullos, peso_prom_capullo_lleno_g, peso_prom_capullo_vacio_g, observaciones) "
                    "VALUES(?,?,?,?,?,?)",
                    (coh_map[sel_coh], str(fecha_c), int(ncap), float(pl), float(pv), obs_c)
                )
            st.success("✅ Cosecha guardada.")

    st.divider()
    st.subheader("Cosechas registradas")
    if cohortes:
        with get_conn() as conn:
            df_c = pd.read_sql_query(
                """
                SELECT l.codigo AS linea, c.campania, cc.fecha, cc.n_capullos,
                       cc.peso_prom_capullo_lleno_g AS pl, cc.peso_prom_capullo_vacio_g AS pv
                FROM cosecha_capullos cc
                JOIN cohorte c ON cc.cohorte_id=c.id
                JOIN linea_bmori l ON c.linea_id=l.id
                ORDER BY cc.fecha DESC
                """,
                conn
            )
        if df_c.empty:
            st.info("Sin datos de cosecha aún.")
        else:
            df_c["shell_ratio"] = df_c.apply(lambda r: shell_ratio(r["pl"], r["pv"]), axis=1)
            st.dataframe(df_c, use_container_width=True)

# ------------- Morera -------------
with tabs[2]:
    st.header("Morera — Fenología y morfometría")

    st.subheader("Registrar variedad")
    v1, v2 = st.columns(2)
    with v1:
        vcod = st.text_input("Código de variedad (único)", placeholder="p.ej. M-01")
        vorig = st.text_input("Origen", placeholder="Banco de germoplasma...")
    with v2:
        vdesc = st.text_area("Descripción", placeholder="Notas de la variedad")

    if st.button("Guardar variedad"):
        if not vcod.strip():
            st.error("El código de variedad es obligatorio.")
        else:
            with get_conn() as conn:
                conn.execute(
                    "INSERT OR IGNORE INTO variedad_morus(codigo, origen, descripcion) VALUES(?,?,?)",
                    (vcod.strip(), vorig.strip(), vdesc.strip())
                )
            st.success("✅ Variedad guardada (o ya existía).")

    st.divider()
    st.subheader("Registrar árbol")
    with get_conn() as conn:
        vars_ = conn.execute("SELECT id, codigo FROM variedad_morus WHERE activo=1 ORDER BY codigo").fetchall()
    if not vars_:
        st.info("Primero registre al menos una variedad de Morus.")
    else:
        vmap = {c: i for i, c in vars_}
        vsel = st.selectbox("Variedad", options=list(vmap.keys()))
        parcela = st.text_input("Parcela", placeholder="Cuadro/Bloque...")
        cod_arbol = st.text_input("Código de árbol", placeholder="ARB-001")
        anio = st.number_input("Año de plantación", min_value=1900, max_value=2100, value=2020, step=1)
        manejo = st.text_input("Manejo", placeholder="Riego/poda/fertilización…")
        aobs = st.text_input("Observaciones", placeholder="Opcional")
        if st.button("Guardar árbol"):
            with get_conn() as conn:
                conn.execute(
                    "INSERT INTO arbol(variedad_id, parcela, codigo_arbol, anio_plantacion, manejo, observaciones) "
                    "VALUES(?,?,?,?,?,?)",
                    (vmap[vsel], parcela, cod_arbol, int(anio), manejo, aobs)
                )
            st.success("✅ Árbol guardado.")

    st.divider()
    st.subheader("Registrar fenología")
    with get_conn() as conn:
        arbs = conn.execute(
            "SELECT a.id, vm.codigo || ' — ' || IFNULL(a.codigo_arbol,'') AS label "
            "FROM arbol a JOIN variedad_morus vm ON a.variedad_id=vm.id ORDER BY a.id DESC"
        ).fetchall()
    if not arbs:
        st.info("Primero registre al menos un árbol.")
    else:
        amap = {label: i for i, label in arbs}
        asel = st.selectbox("Árbol", options=list(amap.keys()))
        fbro = st.date_input("Fecha de brotación", value=dt.date.today())
        fcad = st.date_input("Fecha de caducidad de hojas", value=dt.date.today())
        fobs = st.text_input("Observaciones", placeholder="Opcional")
        if st.button("Guardar fenología"):
            with get_conn() as conn:
                conn.execute(
                    "INSERT INTO fenologia(arbol_id, fecha_brotacion, fecha_caducidad_hojas, observaciones) "
                    "VALUES(?,?,?,?)",
                    (amap[asel], str(fbro), str(fcad), fobs)
                )
            st.success("✅ Fenología guardada.")

    st.divider()
    st.subheader("Registrar morfometría de hoja")
    if not arbs:
        st.info("Primero registre al menos un árbol.")
    else:
        amap2 = {label: i for i, label in arbs}
        asel2 = st.selectbox("Árbol (morfometría)", options=list(amap2.keys()))
        fecha_m = st.date_input("Fecha de muestreo", value=dt.date.today())
        cols = st.columns(3)
        with cols[0]:
            n_ramas = st.number_input("N° de ramas", min_value=0, value=0)
            n_hojas = st.number_input("N° de hojas por rama", min_value=0, value=0)
        with cols[1]:
            largo = st.number_input("Largo hoja (cm)", min_value=0.0, value=0.0, step=0.1)
            ancho = st.number_input("Ancho hoja (cm)", min_value=0.0, value=0.0, step=0.1)
        with cols[2]:
            forma = st.text_input("Forma de hoja", placeholder="entera/lobulada/…")
            peso = st.number_input("Peso de hoja (g)", min_value=0.0, value=0.0, step=0.01)
        mobs = st.text_input("Observaciones", placeholder="Opcional")

        if st.button("Guardar morfometría"):
            with get_conn() as conn:
                conn.execute(
                    "INSERT INTO morfometria_hoja(arbol_id, fecha, n_ramas, n_hojas_por_rama, largo_hoja_cm, ancho_hoja_cm, forma_hoja, peso_hoja_g, observaciones) "
                    "VALUES(?,?,?,?,?,?,?,?,?)",
                    (amap2[asel2], str(fecha_m), int(n_ramas), int(n_hojas), float(largo), float(ancho), forma, float(peso), mobs)
                )
            st.success("✅ Morfometría guardada.")

    st.divider()
    st.subheader("Vistas rápidas de Morera")
    with get_conn() as conn:
        df_var = pd.read_sql_query("SELECT * FROM variedad_morus ORDER BY codigo", conn)
        df_arb = pd.read_sql_query(
            "SELECT a.*, (SELECT codigo FROM variedad_morus WHERE id=a.variedad_id) AS variedad "
            "FROM arbol a ORDER BY a.id DESC", conn
        )
        df_fen = pd.read_sql_query(
            "SELECT f.*, a.codigo_arbol, (SELECT codigo FROM variedad_morus WHERE id=a.variedad_id) AS variedad "
            "FROM fenologia f JOIN arbol a ON f.arbol_id=a.id ORDER BY f.id DESC", conn
        )
        df_mor = pd.read_sql_query(
            "SELECT m.*, a.codigo_arbol, (SELECT codigo FROM variedad_morus WHERE id=a.variedad_id) AS variedad "
            "FROM morfometria_hoja m JOIN arbol a ON m.arbol_id=a.id ORDER BY m.id DESC", conn
        )
    st.write("Variedades"); st.dataframe(df_var if not df_var.empty else pd.DataFrame(), use_container_width=True)
    st.write("Árboles"); st.dataframe(df_arb if not df_arb.empty else pd.DataFrame(), use_container_width=True)
    st.write("Fenología"); st.dataframe(df_fen if not df_fen.empty else pd.DataFrame(), use_container_width=True)
    st.write("Morfometría"); st.dataframe(df_mor if not df_mor.empty else pd.DataFrame(), use_container_width=True)

# ------------- Análisis -------------
with tabs[3]:
    st.header("Análisis — Shell ratio por línea (cosecha)")
    if st.button("Calcular y graficar"):
        with get_conn() as conn:
            df = pd.read_sql_query(
                """
                SELECT l.codigo AS linea, c.campania,
                       cc.peso_prom_capullo_lleno_g AS pl,
                       cc.peso_prom_capullo_vacio_g AS pv
                FROM cosecha_capullos cc
                JOIN cohorte c ON cc.cohorte_id=c.id
                JOIN linea_bmori l ON c.linea_id=l.id
                """,
                conn
            )
        if df.empty:
            st.warning("No hay datos de cosecha para analizar.")
        else:
            df["shell_ratio"] = df.apply(lambda r: shell_ratio(r["pl"], r["pv"]), axis=1)
            st.subheader("Tabla de datos derivados")
            st.dataframe(df, use_container_width=True)

            st.subheader("Boxplot — Shell ratio (%) por línea")
            fig, ax = plt.subplots(figsize=(7, 4.5))
            df_ok = df.dropna(subset=["shell_ratio"]).copy()
            if df_ok.empty:
                st.info("No hay shell_ratio válido para graficar.")
            else:
                df_ok.boxplot(column="shell_ratio", by="linea", ax=ax)
                ax.set_xlabel("Línea")
                ax.set_ylabel("Shell ratio (%)")
                ax.set_title("Distribución de shell ratio por línea")
                plt.suptitle("")
                st.pyplot(fig)

            st.divider()
            st.subheader("ANOVA (opcional)")
            if not HAS_STATSMODELS:
                st.info("statsmodels no está instalado. Si lo desea: pip install statsmodels")
            else:
                model, anova_tbl = anova_shell_por_linea(df)
                if model is None or anova_tbl is None:
                    st.info("Datos insuficientes para ANOVA.")
                else:
                    st.write("Tabla ANOVA (Typ II):")
                    st.dataframe(anova_tbl)
                    with st.expander("Resumen del modelo OLS"):
                        st.text(model.summary().as_text())
            # Guardar en sesión para exportar
            st.session_state["analisis_df"] = df

# ------------- Exportar -------------
with tabs[4]:
    st.header("Exportar a Excel (.xlsx)")
    if "analisis_df" in st.session_state:
        df_exp = st.session_state["analisis_df"]
    else:
        df_exp = pd.DataFrame()

    do_export = st.button("Generar y descargar Excel")
    if do_export:
        if df_exp.empty:
            st.warning("No hay tablas en memoria (ejecute un análisis primero).")
        else:
            from io import BytesIO
            bio = BytesIO()
            with pd.ExcelWriter(bio, engine="openpyxl") as xw:
                df_exp.to_excel(xw, index=False, sheet_name="Shell_ratio")
            bio.seek(0)
            st.download_button(
                "Descargar sericultura_resultados.xlsx",
                data=bio.read(),
                file_name="sericultura_resultados.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

st.caption("Versión de demostración — base SQLite local: sericultura.db")
