import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy import linalg
from scipy.linalg import svd, qr, eig, inv, det, norm, kron, expm
import re
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURACIÓN DE LA PÁGINA
# ============================================================
st.set_page_config(
    page_title="🧮 Calculadora de Álgebra Lineal",
    page_icon="🧮",
    layout="wide"
)

st.title("🧮 Calculadora de Álgebra Lineal - COMPLETA")
st.markdown("---")

# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

def texto_a_matriz(texto):
    """Convierte texto a matriz numpy"""
    if not texto:
        return None
    try:
        filas = [f.strip() for f in texto.split(';') if f.strip()]
        matriz = []
        for fila in filas:
            elementos = [float(x.strip()) for x in fila.split(',') if x.strip()]
            if elementos:
                matriz.append(elementos)
        if not matriz:
            return None
        return np.array(matriz)
    except:
        return None

def texto_a_vector(texto):
    """Convierte texto a vector numpy"""
    if not texto:
        return None
    try:
        elementos = [float(x.strip()) for x in texto.split(',') if x.strip()]
        if len(elementos) < 2:
            return None
        return np.array(elementos)
    except:
        return None

def formatear_matriz(mat, decimales=6):
    """Formatea matriz para display"""
    if mat is None:
        return "No disponible"
    if isinstance(mat, str):
        return mat
    mat = mat.copy()
    mat[np.abs(mat) < 1e-10] = 0
    return np.round(mat, decimales)

def limpiar_ceros(mat, tol=1e-10):
    """Limpia ceros de la matriz"""
    if mat is None:
        return None
    mat = mat.copy()
    mat[np.abs(mat) < tol] = 0
    return mat

def matriz_a_texto(mat):
    """Convierte matriz a texto para display"""
    if mat is None:
        return ""
    mat = limpiar_ceros(mat)
    return np.array2string(mat, precision=6, suppress_small=True)

# ============================================================
# FUNCIONES AVANZADAS DE ÁLGEBRA LINEAL
# ============================================================

def resolver_sistema(A, b):
    """Resuelve sistema Ax = b"""
    try:
        if A.shape[0] != len(b):
            return None, "Error: Dimensiones incompatibles"
        
        rango_A = np.linalg.matrix_rank(A)
        rango_AB = np.linalg.matrix_rank(np.column_stack([A, b]))
        
        if rango_A != rango_AB:
            return None, "Sistema INCOMPATIBLE (no tiene solución)"
        
        if A.shape[1] > rango_A:
            sol = np.linalg.lstsq(A, b, rcond=None)[0]
            return sol, "Sistema COMPATIBLE INDETERMINADO (infinitas soluciones)"
        
        sol = np.linalg.solve(A, b)
        return sol, "Sistema COMPATIBLE DETERMINADO (solución única)"
    except Exception as e:
        return None, f"Error: {str(e)}"

def minimos_cuadrados(A, b):
    """Solución por mínimos cuadrados"""
    try:
        sol = np.linalg.lstsq(A, b, rcond=None)[0]
        residuos = b - A @ sol
        return sol, residuos, "Solución por mínimos cuadrados"
    except Exception as e:
        return None, None, f"Error: {str(e)}"

def gram_schmidt(vectores):
    """Ortogonalización de Gram-Schmidt"""
    try:
        n = vectores.shape[1]
        base_ortogonal = np.zeros_like(vectores)
        base_ortonormal = np.zeros_like(vectores)
        
        for i in range(n):
            v = vectores[:, i].copy()
            for j in range(i):
                if np.sum(base_ortogonal[:, j]**2) > 0:
                    v = v - (np.dot(v, base_ortogonal[:, j]) / np.sum(base_ortogonal[:, j]**2)) * base_ortogonal[:, j]
            base_ortogonal[:, i] = v
            if np.sum(v**2) > 0:
                base_ortonormal[:, i] = v / np.sqrt(np.sum(v**2))
        
        return base_ortogonal, base_ortonormal, "Gram-Schmidt completado"
    except Exception as e:
        return None, None, f"Error: {str(e)}"

def dependencia_lineal(vectores):
    """Verifica dependencia lineal"""
    try:
        rango = np.linalg.matrix_rank(vectores)
        n = vectores.shape[1]
        if rango < n:
            return True, rango, n, f"Linealmente DEPENDIENTES (rango = {rango} de {n})"
        else:
            return False, rango, n, f"Linealmente INDEPENDIENTES (rango = {rango} de {n})"
    except Exception as e:
        return None, None, None, f"Error: {str(e)}"

def kernel_imagen(A):
    """Calcula kernel e imagen"""
    try:
        u, s, vh = svd(A)
        tol = 1e-10
        rango = np.sum(s > tol)
        kernel = vh[rango:].T
        
        imagen = A[:, :rango]
        
        return kernel, imagen, rango, "Kernel e Imagen calculados"
    except Exception as e:
        return None, None, None, f"Error: {str(e)}"

def matriz_hilbert(n):
    """Genera matriz de Hilbert"""
    H = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            H[i, j] = 1 / (i + j + 1)
    return H

def matriz_vandermonde(x):
    """Genera matriz de Vandermonde"""
    n = len(x)
    V = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            V[i, j] = x[i] ** j
    return V

def verificar_propiedades(A):
    """Verifica propiedades de la matriz"""
    resultados = {}
    
    if A.shape[0] == A.shape[1]:
        # Simetría
        resultados['simetrica'] = np.allclose(A, A.T, atol=1e-10)
        
        # Definida positiva
        try:
            eigen_vals = np.linalg.eigvals(A)
            resultados['definida_positiva'] = np.all(eigen_vals > 0)
            resultados['definida_negativa'] = np.all(eigen_vals < 0)
        except:
            resultados['definida_positiva'] = False
            resultados['definida_negativa'] = False
        
        # Ortogonalidad
        resultados['ortogonal'] = np.allclose(A @ A.T, np.eye(A.shape[0]), atol=1e-10)
        
        # Normalidad
        resultados['normal'] = np.allclose(A @ A.T - A.T @ A, np.zeros_like(A), atol=1e-10)
        
        # Diagonal dominante
        diag = np.abs(np.diag(A))
        off_diag = np.sum(np.abs(A), axis=1) - diag
        resultados['diagonal_dominante'] = np.all(2*diag > off_diag)
        
        # Matriz de Markov
        resultados['markov'] = np.all(A >= 0) and np.allclose(np.sum(A, axis=1), 1, atol=1e-10)
        
        # Matriz de permutación
        if np.all(np.isin(A, [0, 1])):
            resultados['permutacion'] = np.all(np.sum(A, axis=0) == 1) and np.all(np.sum(A, axis=1) == 1)
        else:
            resultados['permutacion'] = False
        
        # Matriz diagonal
        resultados['diagonal'] = np.allclose(A - np.diag(np.diag(A)), np.zeros_like(A), atol=1e-10)
    
    return resultados

def ajuste_polinomial(x, y, grado):
    """Ajuste polinomial por mínimos cuadrados"""
    try:
        if len(x) != len(y):
            return None, "Error: x e y deben tener la misma longitud"
        if grado < 1:
            return None, "Error: El grado debe ser >= 1"
        if len(x) <= grado:
            return None, "Error: Se necesitan más datos que el grado del polinomio"
        
        X = np.zeros((len(x), grado + 1))
        for i in range(grado + 1):
            X[:, i] = x ** i
        
        coef = np.linalg.lstsq(X, y, rcond=None)[0]
        y_pred = X @ coef
        r2 = 1 - np.sum((y - y_pred)**2) / np.sum((y - np.mean(y))**2)
        
        return coef, y_pred, r2, f"Ajuste polinomial grado {grado} completado"
    except Exception as e:
        return None, None, None, f"Error: {str(e)}"

def analisis_pca(X):
    """Análisis de Componentes Principales"""
    try:
        X_cent = X - np.mean(X, axis=0)
        S = np.cov(X_cent, rowvar=False)
        eigen_vals, eigen_vecs = np.linalg.eig(S)
        idx = np.argsort(eigen_vals)[::-1]
        eigen_vals = eigen_vals[idx]
        eigen_vecs = eigen_vecs[:, idx]
        componentes = X_cent @ eigen_vecs
        varianza_explicada = eigen_vals / np.sum(eigen_vals)
        
        return componentes, varianza_explicada, eigen_vecs, "PCA completado"
    except Exception as e:
        return None, None, None, f"Error: {str(e)}"

# ============================================================
# INTERFAZ DE USUARIO
# ============================================================

# Crear pestañas
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "📐 Matrices",
    "📏 Vectores",
    "🔢 Sistemas",
    "🔄 Transformaciones",
    "🎯 Matrices Especiales",
    "✅ Propiedades",
    "🎲 Aleatoria",
    "📖 Guía"
])

# ============================================================
# PESTAÑA 1: MATRICES
# ============================================================

with tab1:
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("📊 Matriz A")
        matriz_a_text = st.text_area(
            "Ingresa Matriz A (filas separadas por ;, elementos por ,):",
            value="1,2,3;4,5,6;7,8,9",
            height=100,
            key="mat_a_text"
        )
        
        st.subheader("📊 Matriz B (opcional)")
        matriz_b_text = st.text_area(
            "Ingresa Matriz B:",
            value="9,8,7;6,5,4;3,2,1",
            height=100,
            key="mat_b_text"
        )
        
        escalar = st.number_input("Valor escalar:", value=2.0, step=0.5, key="escalar_mat")
        
        if st.button("🔍 Calcular Matrices", key="btn_matrices"):
            A = texto_a_matriz(matriz_a_text)
            B = texto_a_matriz(matriz_b_text)
            
            if A is None:
                st.error("❌ Error: Matriz A inválida")
            else:
                st.session_state['mat_A'] = A
                st.session_state['mat_B'] = B
                st.session_state['escalar_val'] = escalar
                st.success("✅ Cálculo completado")
    
    with col2:
        if 'mat_A' in st.session_state:
            A = st.session_state['mat_A']
            B = st.session_state['mat_B']
            escalar = st.session_state.get('escalar_val', 2.0)
            
            st.subheader("📊 Resultados")
            
            st.write("**Matriz A:**")
            st.code(matriz_a_texto(A))
            
            if B is not None:
                st.write("**Matriz B:**")
                st.code(matriz_a_texto(B))
            
            st.write("**Operaciones:**")
            
            st.write("**Transpuesta:**")
            st.code(matriz_a_texto(A.T))
            
            if A.shape[0] == A.shape[1]:
                st.write(f"**Traza:** {np.trace(A):.6f}")
                st.write(f"**Determinante:** {np.linalg.det(A):.6f}")
                
                try:
                    inv_A = np.linalg.inv(A)
                    st.write("**Inversa:**")
                    st.code(matriz_a_texto(inv_A))
                except:
                    st.write("**Inversa:** Singular (no invertible)")
            
            st.write(f"**Rango:** {np.linalg.matrix_rank(A)}")
            st.write(f"**Norma (Frobenius):** {np.linalg.norm(A, 'fro'):.6f}")
            
            if A.shape[0] == A.shape[1]:
                st.write(f"**Condición:** {np.linalg.cond(A):.6f}")
            
            st.write(f"**Producto por escalar ({escalar}):**")
            st.code(matriz_a_texto(escalar * A))
            
            if B is not None:
                if A.shape == B.shape:
                    st.write("**Suma A+B:**")
                    st.code(matriz_a_texto(A + B))
                    st.write("**Resta A-B:**")
                    st.code(matriz_a_texto(A - B))
                    st.write("**Producto de Hadamard:**")
                    st.code(matriz_a_texto(A * B))
                
                if A.shape[1] == B.shape[0]:
                    st.write("**Multiplicación A×B:**")
                    st.code(matriz_a_texto(A @ B))
                
                st.write("**Producto de Kronecker:**")
                st.code(matriz_a_texto(np.kron(A, B)))

# ============================================================
# PESTAÑA 2: VECTORES
# ============================================================

with tab2:
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("📊 Vector u")
        vector_u_text = st.text_area(
            "Ingresa Vector u (elementos separados por ,):",
            value="1,2,3",
            height=60,
            key="vec_u_text"
        )
        
        st.subheader("📊 Vector v")
        vector_v_text = st.text_area(
            "Ingresa Vector v:",
            value="4,5,6",
            height=60,
            key="vec_v_text"
        )
        
        escalar_vec = st.number_input("Valor escalar:", value=2.0, step=0.5, key="escalar_vec")
        
        if st.button("🔍 Calcular Vectores", key="btn_vectores"):
            u = texto_a_vector(vector_u_text)
            v = texto_a_vector(vector_v_text)
            
            if u is None:
                st.error("❌ Error: Vector u inválido")
            else:
                st.session_state['vec_u'] = u
                st.session_state['vec_v'] = v
                st.session_state['escalar_vec_val'] = escalar_vec
                st.success("✅ Cálculo completado")
    
    with col2:
        if 'vec_u' in st.session_state:
            u = st.session_state['vec_u']
            v = st.session_state['vec_v']
            escalar_vec = st.session_state.get('escalar_vec_val', 2.0)
            
            st.subheader("📊 Resultados")
            
            st.write(f"**u =** {u.tolist()}")
            if v is not None:
                st.write(f"**v =** {v.tolist()}")
            
            st.write("**Operaciones:**")
            
            st.write(f"**Norma de u:** {np.linalg.norm(u):.6f}")
            if v is not None:
                st.write(f"**Norma de v:** {np.linalg.norm(v):.6f}")
            
            if np.linalg.norm(u) > 0:
                st.write(f"**Vector unitario de u:** {u/np.linalg.norm(u)}")
            
            st.write(f"**Producto por escalar ({escalar_vec}):** {escalar_vec * u}")
            
            if v is not None and len(u) == len(v):
                st.write(f"**Suma u+v:** {u + v}")
                st.write(f"**Resta u-v:** {u - v}")
                st.write(f"**Producto escalar (u·v):** {np.dot(u, v):.6f}")
                
                if np.linalg.norm(u) > 0 and np.linalg.norm(v) > 0:
                    cos_theta = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
                    cos_theta = np.clip(cos_theta, -1, 1)
                    angulo = np.arccos(cos_theta)
                    st.write(f"**Ángulo entre u y v:** {angulo:.6f} rad ({angulo*180/np.pi:.6f}°)")
                
                if np.linalg.norm(v) > 0:
                    proy = (np.dot(u, v) / np.linalg.norm(v)**2) * v
                    st.write(f"**Proyección de u sobre v:** {proy}")
                
                if len(u) == 3 and len(v) == 3:
                    cross = np.cross(u, v)
                    st.write(f"**Producto vectorial (u×v):** {cross}")
                
                matriz_vect = np.column_stack([u, v])
                dep, rango, dim, msg = dependencia_lineal(matriz_vect)
                st.write(f"**Dependencia Lineal:** {msg}")
                
                ortog, orton, msg_gs = gram_schmidt(matriz_vect)
                if ortog is not None:
                    st.write("**Gram-Schmidt (Base Ortogonal):**")
                    st.code(matriz_a_texto(ortog))
                    st.write("**Gram-Schmidt (Base Ortonormal):**")
                    st.code(matriz_a_texto(orton))

# ============================================================
# PESTAÑA 3: SISTEMAS
# ============================================================

with tab3:
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("📊 Matriz A (coeficientes)")
        sistema_A_text = st.text_area(
            "Ingresa Matriz A:",
            value="1,2;3,4",
            height=80,
            key="sist_A_text"
        )
        
        st.subheader("📊 Vector b (términos independientes)")
        sistema_b_text = st.text_area(
            "Ingresa Vector b:",
            value="5,6",
            height=60,
            key="sist_b_text"
        )
        
        if st.button("🔍 Resolver Sistema", key="btn_sistemas"):
            A = texto_a_matriz(sistema_A_text)
            b = texto_a_vector(sistema_b_text)
            
            if A is None or b is None:
                st.error("❌ Error: Matriz A o vector b inválidos")
            else:
                st.session_state['sist_A'] = A
                st.session_state['sist_b'] = b
                st.success("✅ Sistema resuelto")
    
    with col2:
        if 'sist_A' in st.session_state:
            A = st.session_state['sist_A']
            b = st.session_state['sist_b']
            
            st.subheader("📊 Resultados")
            
            st.write("**Matriz A:**")
            st.code(matriz_a_texto(A))
            st.write("**Vector b:**")
            st.write(b.tolist())
            
            sol, msg = resolver_sistema(A, b)
            st.write(f"**Sistema Ax = b:**")
            st.write(f"  {msg}")
            if sol is not None:
                st.write(f"  **Solución:** {sol}")
            
            sol_mc, residuos, msg_mc = minimos_cuadrados(A, b)
            st.write(f"**Mínimos Cuadrados:**")
            st.write(f"  {msg_mc}")
            if sol_mc is not None:
                st.write(f"  **Solución:** {sol_mc}")
                st.write(f"  **Residuos:** {residuos}")

# ============================================================
# PESTAÑA 4: TRANSFORMACIONES
# ============================================================

with tab4:
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("📊 Matriz de Transformación")
        trans_A_text = st.text_area(
            "Ingresa Matriz A:",
            value="1,2,3;4,5,6;7,8,9",
            height=100,
            key="trans_A_text"
        )
        
        st.subheader("📊 Vector de entrada (opcional)")
        trans_v_text = st.text_area(
            "Ingresa Vector v:",
            value="1,2,3",
            height=60,
            key="trans_v_text"
        )
        
        if st.button("🔍 Calcular Transformaciones", key="btn_transformaciones"):
            A = texto_a_matriz(trans_A_text)
            v = texto_a_vector(trans_v_text)
            
            if A is None:
                st.error("❌ Error: Matriz inválida")
            else:
                st.session_state['trans_A'] = A
                st.session_state['trans_v'] = v
                st.success("✅ Transformaciones calculadas")
    
    with col2:
        if 'trans_A' in st.session_state:
            A = st.session_state['trans_A']
            v = st.session_state['trans_v']
            
            st.subheader("📊 Resultados")
            
            st.write("**Matriz de Transformación:**")
            st.code(matriz_a_texto(A))
            
            kernel, imagen, rango, msg = kernel_imagen(A)
            st.write(f"**{msg}**")
            st.write(f"**Rango:** {rango}")
            
            if kernel is not None:
                st.write("**Kernel (Espacio Nulo):**")
                if kernel.size == 0 or kernel.shape[1] == 0:
                    st.write("  Solo el vector cero (trivial)")
                else:
                    st.code(matriz_a_texto(kernel))
            
            if imagen is not None:
                st.write("**Imagen (Rango):**")
                st.code(matriz_a_texto(imagen))
            
            if A.shape[0] == A.shape[1]:
                if np.linalg.matrix_rank(A) == A.shape[1]:
                    st.write("**Inyectiva:** Sí (matriz invertible)")
                    st.write("**Sobreyectiva:** Sí (matriz invertible)")
                    st.write("**Biyectiva:** Sí (isomorfismo)")
                else:
                    st.write("**Inyectiva:** No")
                    st.write("**Sobreyectiva:** No")
                    st.write("**Biyectiva:** No")
            else:
                iny = "Sí" if np.linalg.matrix_rank(A) == A.shape[1] else "No"
                sob = "Sí" if np.linalg.matrix_rank(A) == A.shape[0] else "No"
                st.write(f"**Inyectiva:** {iny}")
                st.write(f"**Sobreyectiva:** {sob}")
                st.write(f"**Biyectiva:** {'Sí' if iny == 'Sí' and sob == 'Sí' else 'No'}")
            
            if v is not None:
                if len(v) == A.shape[1]:
                    st.write("**Transformación aplicada a v:**")
                    st.write(A @ v)
                else:
                    st.write("**Transformación aplicada a v:** Dimensiones incompatibles")

# ============================================================
# PESTAÑA 5: MATRICES ESPECIALES
# ============================================================

with tab5:
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("📊 Matriz A (para funciones)")
        esp_A_text = st.text_area(
            "Ingresa Matriz A:",
            value="1,2;3,4",
            height=80,
            key="esp_A_text"
        )
        
        st.subheader("📊 Tamaño para matrices especiales")
        n_esp = st.number_input("Tamaño (n):", min_value=2, max_value=10, value=4, step=1, key="n_esp")
        
        if st.button("🔍 Calcular Matrices Especiales", key="btn_especiales"):
            A = texto_a_matriz(esp_A_text)
            st.session_state['esp_A'] = A
            st.session_state['n_esp'] = n_esp
            st.success("✅ Matrices especiales generadas")
    
    with col2:
        if 'esp_A' in st.session_state:
            A = st.session_state['esp_A']
            n = st.session_state['n_esp']
            
            st.subheader("📊 Resultados")
            
            H = matriz_hilbert(n)
            st.write(f"**Matriz de Hilbert ({n}×{n}):**")
            st.code(matriz_a_texto(H))
            
            x = np.arange(1, n+1)
            V = matriz_vandermonde(x)
            st.write(f"**Matriz de Vandermonde ({n}×{n}):**")
            st.code(matriz_a_texto(V))
            
            if A is not None and A.shape[0] == A.shape[1]:
                try:
                    exp_A = linalg.expm(A)
                    st.write("**Exponencial de matriz:**")
                    st.code(matriz_a_texto(exp_A))
                except:
                    st.write("**Exponencial de matriz:** Error en el cálculo")
            
            if A is not None and A.shape[0] >= 3:
                x = np.arange(1, A.shape[0]+1)
                y = A[:, 0]
                for g in range(1, 4):
                    coef, y_pred, r2, msg = ajuste_polinomial(x, y, g)
                    if coef is not None:
                        st.write(f"**Ajuste polinomial grado {g}:**")
                        st.write(f"  R² = {r2:.6f}")
                        st.write(f"  Coeficientes: {coef}")

# ============================================================
# PESTAÑA 6: PROPIEDADES
# ============================================================

with tab6:
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("📊 Matriz para propiedades")
        prop_A_text = st.text_area(
            "Ingresa Matriz A:",
            value="1,2,3;4,5,6;7,8,9",
            height=100,
            key="prop_A_text"
        )
        
        st.subheader("📊 Datos para PCA")
        pca_data_text = st.text_area(
            "Ingresa datos (filas=observaciones, columnas=variables):",
            value="1,2,3;4,5,6;7,8,9",
            height=80,
            key="pca_data_text"
        )
        
        if st.button("🔍 Calcular Propiedades", key="btn_propiedades"):
            A = texto_a_matriz(prop_A_text)
            pca_data = texto_a_matriz(pca_data_text)
            
            if A is None:
                st.error("❌ Error: Matriz inválida")
            else:
                st.session_state['prop_A'] = A
                st.session_state['pca_data'] = pca_data
                st.success("✅ Propiedades calculadas")
    
    with col2:
        if 'prop_A' in st.session_state:
            A = st.session_state['prop_A']
            pca_data = st.session_state['pca_data']
            
            st.subheader("📊 Resultados")
            
            st.write("**Matriz:**")
            st.code(matriz_a_texto(A))
            
            props = verificar_propiedades(A)
            st.write("**Propiedades:**")
            for key, value in props.items():
                icon = "✅" if value else "❌"
                st.write(f"  {icon} {key.replace('_', ' ').title()}: {'Sí' if value else 'No'}")
            
            if pca_data is not None and pca_data.shape[0] > 1 and pca_data.shape[1] > 1:
                componentes, var_exp, loadings, msg = analisis_pca(pca_data)
                if componentes is not None:
                    st.write("**Análisis de Componentes Principales (PCA):**")
                    st.write(f"  {msg}")
                    st.write("**Varianza explicada:**")
                    for i, v in enumerate(var_exp):
                        st.write(f"  Componente {i+1}: {v*100:.2f}%")
                    st.write("**Loadings:**")
                    st.code(matriz_a_texto(loadings))

# ============================================================
# PESTAÑA 7: MATRIZ ALEATORIA
# ============================================================

with tab7:
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("🎲 Generar Matriz Aleatoria")
        n_filas_rand = st.number_input("Filas:", min_value=1, max_value=10, value=3, step=1, key="n_filas_rand")
        n_columnas_rand = st.number_input("Columnas:", min_value=1, max_value=10, value=3, step=1, key="n_columnas_rand")
        
        if st.button("🎲 Generar", key="btn_random"):
            A = np.random.randn(n_filas_rand, n_columnas_rand)
            st.session_state['rand_A'] = A
            
            B = np.random.randn(n_filas_rand, 3)
            st.session_state['rand_B'] = B
            st.success("🎲 Matriz aleatoria generada")
    
    with col2:
        if 'rand_A' in st.session_state:
            A = st.session_state['rand_A']
            B = st.session_state['rand_B']
            
            st.subheader("📊 Resultados")
            
            st.write("**Matriz A:**")
            st.code(matriz_a_texto(A))
            
            st.write("**Matriz B:**")
            st.code(matriz_a_texto(B))
            
            st.write("**Operaciones:**")
            
            try:
                concat = np.column_stack([A, B])
                st.write("**Concatenar (A|B):**")
                st.code(matriz_a_texto(concat))
            except:
                st.write("**Concatenar (A|B):** Error en dimensiones")
            
            try:
                apilar = np.row_stack([A, B])
                st.write("**Apilar (A/B):**")
                st.code(matriz_a_texto(apilar))
            except:
                st.write("**Apilar (A/B):** Error en dimensiones")
            
            total = A.size
            n_nuevo = int(np.round(np.sqrt(total)))
            if n_nuevo * n_nuevo == total:
                reshape = A.reshape(n_nuevo, n_nuevo)
                st.write(f"**Redimensionar ({n_nuevo}×{n_nuevo}):**")
                st.code(matriz_a_texto(reshape))
            else:
                st.write("**Redimensionar:** No se puede redimensionar")

# ============================================================
# PESTAÑA 8: GUÍA
# ============================================================

with tab8:
    st.markdown(
        """
        # 📖 Guía de Uso - Álgebra Lineal
        
        ## 📊 Formatos de Ingreso
        
        ### Matrices
        - Filas separadas por `;`
        - Elementos separados por `,`
        - Ejemplo: `1,2,3;4,5,6;7,8,9`
        
        ### Vectores
        - Elementos separados por `,`
        - Ejemplo: `1,2,3`
        
        ## 📌 Todas las Operaciones Disponibles
        
        ### Matrices
        - ✅ Transpuesta, Hermitiana
        - ✅ Traza, Determinante, Inversa
        - ✅ Rango, Norma, Condición
        - ✅ Suma, Resta, Multiplicación
        - ✅ Producto por escalar, Potencia
        - ✅ Producto de Hadamard
        - ✅ Producto de Kronecker
        
        ### Vectores
        - ✅ Suma, Resta, Producto escalar
        - ✅ Producto vectorial (R³)
        - ✅ Norma, Vector unitario
        - ✅ Ángulo, Proyección
        - ✅ Dependencia Lineal
        - ✅ Gram-Schmidt (ortogonalización)
        
        ### Sistemas
        - ✅ Solución Ax=b
        - ✅ Mínimos cuadrados
        - ✅ Compatibilidad
        
        ### Transformaciones
        - ✅ Kernel (Núcleo)
        - ✅ Imagen (Rango)
        - ✅ Inyectividad/Sobreyectividad
        - ✅ Cambio de base
        - ✅ Aplicar a vector
        
        ### Matrices Especiales
        - ✅ Hilbert
        - ✅ Vandermonde
        - ✅ Exponencial de matriz
        - ✅ Ajuste polinomial
        
        ### Propiedades
        - ✅ Simetría, Ortogonalidad
        - ✅ Definida Positiva/Negativa
        - ✅ Diagonal Dominante
        - ✅ Matriz de Markov
        - ✅ Matriz de Permutación
        - ✅ Análisis PCA
        """
    )
