import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.animation import FuncAnimation
import matplotlib
matplotlib.use('Agg')
import time
import base64
from io import BytesIO

# Configuración de la página
st.set_page_config(
    page_title="Calculadora de Cadenas de Markov",
    page_icon="🔄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS personalizados
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .st-emotion-cache-18ni7ap {
        background-color: #ffffff;
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    h1, h2, h3 {
        color: #2c3e50;
    }
    .stButton>button {
        background-color: #3498db;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 0.5rem 1rem;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #2980b9;
        transform: translateY(-2px);
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .card {
        background-color: white;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        margin-bottom: 20px;
    }
    .success-msg {
        padding: 10px 15px;
        border-radius: 5px;
        background-color: #d4edda;
        color: #155724;
        margin: 10px 0;
        animation: fadeIn 0.5s ease-in;
    }
    @keyframes fadeIn {
        0% { opacity: 0; }
        100% { opacity: 1; }
    }
    .matrix-input td {
        padding: 5px;
    }
    /* Colores para el heatmap */
    .heatmap-cool {
        background: linear-gradient(to right, #74ebd5, #ACB6E5);
    }
    .animate-panel {
        animation: slideIn 0.5s ease-out;
    }
    @keyframes slideIn {
        0% { transform: translateY(20px); opacity: 0; }
        100% { transform: translateY(0); opacity: 1; }
    }
</style>
""", unsafe_allow_html=True)

# Título y descripción de la aplicación
st.markdown("<h1 style='text-align: center;'>Calculadora de Cadenas de Markov</h1>", unsafe_allow_html=True)

with st.container():
    st.markdown("""
    <div class="card animate-panel">
        <p>Esta calculadora te permite trabajar con cadenas de Markov, incluyendo:</p>
        <ul>
            <li>Cálculo de probabilidades de estado futuras</li>
            <li>Análisis de estados absorbentes</li>
            <li>Visualización de evolución de estados</li>
            <li>Tiempo medio hasta la absorción</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# Clase para manejar las operaciones de cadenas de Markov
class MarkovChain:
    def __init__(self, transition_matrix, states=None):
        """
        Inicializa una cadena de Markov con matriz de transición y estados.
        
        Args:
            transition_matrix: Matriz numpy o lista de listas con probabilidades de transición
            states: Nombres de los estados (opcional)
        """
        self.P = np.array(transition_matrix, dtype=float)
        self.n_states = self.P.shape[0]
        
        if states is None:
            self.states = [f"Estado {i+1}" for i in range(self.n_states)]
        else:
            self.states = states
            
        # Validar que la matriz sea estocástica
        row_sums = np.sum(self.P, axis=1)
        if not np.allclose(row_sums, np.ones(self.n_states)):
            raise ValueError("La matriz de transición debe ser estocástica (cada fila debe sumar 1)")
    
    def get_state_distribution(self, initial_state, n_steps):
        """
        Calcula la distribución de probabilidad después de n pasos.
        
        Args:
            initial_state: Vector de estado inicial o índice del estado
            n_steps: Número de pasos a proyectar
            
        Returns:
            Vector de probabilidad después de n pasos
        """
        if isinstance(initial_state, int):
            # Si se da un índice, crear vector con 1 en esa posición
            state = np.zeros(self.n_states)
            state[initial_state] = 1.0
        else:
            # Usar el vector proporcionado
            state = np.array(initial_state)
            
        # Calcular la distribución después de n pasos
        return np.linalg.matrix_power(self.P, n_steps) @ state
    
    def get_state_evolution(self, initial_state, n_steps):
        """
        Calcula la evolución de la distribución de estados para cada paso.
        
        Args:
            initial_state: Vector de estado inicial o índice del estado
            n_steps: Número de pasos a proyectar
            
        Returns:
            Matriz donde cada fila es la distribución después de i pasos
        """
        if isinstance(initial_state, int):
            state = np.zeros(self.n_states)
            state[initial_state] = 1.0
        else:
            state = np.array(initial_state)
            
        evolution = np.zeros((n_steps+1, self.n_states))
        evolution[0] = state
        
        for i in range(1, n_steps+1):
            evolution[i] = self.P @ evolution[i-1]
            
        return evolution
    
    def is_absorbing(self):
        """
        Verifica si la cadena de Markov tiene estados absorbentes.
        
        Returns:
            True si hay al menos un estado absorbente, False en caso contrario
        """
        for i in range(self.n_states):
            if self.P[i, i] == 1:
                return True
        return False
    
    def get_absorbing_states(self):
        """
        Identifica los estados absorbentes en la cadena.
        
        Returns:
            Lista de índices de estados absorbentes
        """
        return [i for i in range(self.n_states) if self.P[i, i] == 1]
    
    def canonical_form(self):
        """
        Reorganiza la matriz de transición en forma canónica para cadenas absorbentes.
        
        Returns:
            Diccionario con información sobre la forma canónica
        """
        absorbing = self.get_absorbing_states()
        non_absorbing = [i for i in range(self.n_states) if i not in absorbing]
        
        if not absorbing:
            return None
        
        # Reordenar la matriz para tener estados absorbentes primero
        order = absorbing + non_absorbing
        P_canonical = self.P[np.ix_(order, order)]
        
        # Extraer submatrices
        t = len(absorbing)
        Q = P_canonical[t:, t:]  # Transiciones entre estados no absorbentes
        R = P_canonical[t:, :t]  # Transiciones de no absorbentes a absorbentes
        
        return {
            "order": order,
            "states_ordered": [self.states[i] for i in order],
            "absorbing": absorbing,
            "non_absorbing": non_absorbing,
            "P_canonical": P_canonical,
            "Q": Q,
            "R": R
        }
    
    def absorption_probabilities(self):
        """
        Calcula las probabilidades de absorción para estados absorbentes.
        
        Returns:
            Matriz B donde B_ij es la probabilidad de ser absorbido en el estado j
            partiendo del estado no absorbente i
        """
        cf = self.canonical_form()
        if cf is None:
            return None
        
        # Matriz fundamental
        I = np.eye(len(cf["non_absorbing"]))
        try:
            N = np.linalg.inv(I - cf["Q"])
            # Probabilidades de absorción
            B = N @ cf["R"]
            return B
        except np.linalg.LinAlgError:
            st.error("Error al calcular la matriz inversa. La cadena podría no tener una solución única.")
            return None
    
    def expected_steps_to_absorption(self):
        """
        Calcula el número esperado de pasos hasta la absorción.
        
        Returns:
            Vector con el número esperado de pasos para cada estado no absorbente
        """
        cf = self.canonical_form()
        if cf is None:
            return None
        
        # Matriz fundamental
        I = np.eye(len(cf["non_absorbing"]))
        try:
            N = np.linalg.inv(I - cf["Q"])
            # Número esperado de pasos
            t = N @ np.ones(len(cf["non_absorbing"]))
            return t
        except np.linalg.LinAlgError:
            st.error("Error al calcular la matriz inversa. La cadena podría no tener una solución única.")
            return None
    
    def steady_state(self):
        """
        Calcula la distribución estacionaria de la cadena de Markov (si existe).
        
        Returns:
            Vector de distribución estacionaria o None si no existe una única
        """
        # Para cadenas absorbentes, no hay una única distribución estacionaria
        if self.is_absorbing():
            return None
            
        # Resolver el sistema de ecuaciones π = πP
        A = np.eye(self.n_states) - self.P.T
        A[-1] = np.ones(self.n_states)  # Reemplazar última fila con restricción de suma
        b = np.zeros(self.n_states)
        b[-1] = 1.0  # La suma de probabilidades debe ser 1
        
        try:
            pi = np.linalg.solve(A, b)
            return pi if np.all(pi >= 0) else None
        except np.linalg.LinAlgError:
            return None

# Sidebar para configuración
with st.sidebar:
    st.markdown("<h3>Configuración</h3>", unsafe_allow_html=True)
    
    # Número de estados
    num_states = st.number_input("Número de estados", min_value=2, max_value=10, value=3, step=1)
    
    # Opción para nombrar estados
    custom_states = st.checkbox("Personalizar nombres de estados", value=False)
    
    state_names = []
    if custom_states:
        for i in range(num_states):
            default_name = f"Estado {i+1}"
            state_names.append(st.text_input(f"Nombre del estado {i+1}", value=default_name))
    else:
        state_names = [f"Estado {i+1}" for i in range(num_states)]
    
    st.markdown("---")
    
    # Selección de tema de color
    color_theme = st.selectbox(
        "Tema de color",
        ["Azul", "Verde", "Púrpura", "Naranja"]
    )
    
    # Mapeo de temas de color
    color_maps = {
        "Azul": "Blues",
        "Verde": "Greens",
        "Púrpura": "Purples",
        "Naranja": "Oranges"
    }
    
    selected_colormap = color_maps[color_theme]
    
    st.markdown("---")
    
    # Opciones avanzadas
    with st.expander("Opciones avanzadas"):
        animation_speed = st.slider("Velocidad de animación", 0.1, 2.0, 1.0, 0.1)
        decimal_places = st.slider("Decimales a mostrar", 2, 6, 4, 1)

# Función para crear una matriz de input
def create_matrix_input(num_rows, num_cols, key_prefix, default_values=None):
    matrix = []
    
    # CSS personalizado para la matriz
    st.markdown("""
    <style>
        .matrix-container {
            padding: 15px;
            border-radius: 8px;
            background-color: white;
            box-shadow: 0 2px 5px rgba(0,0,0,0.05);
            animation: fadeIn 0.5s ease;
        }
        .matrix-row {
            display: flex;
            justify-content: center;
            margin-bottom: 8px;
        }
        .matrix-cell {
            width: 80px;
            margin: 0 5px;
        }
        input[type=number] {
            text-align: center;
        }
    </style>
    """, unsafe_allow_html=True)
    
    cols = st.columns([0.1, 0.8, 0.1])
    
    with cols[1]:
        st.markdown("<div class='matrix-container'>", unsafe_allow_html=True)
        
        # Encabezados de columnas
        header_cols = st.columns(num_cols+1)
        with header_cols[0]:
            st.write("")
        for j in range(num_cols):
            with header_cols[j+1]:
                st.markdown(f"<div style='text-align: center; font-weight: bold;'>{state_names[j]}</div>", 
                          unsafe_allow_html=True)
        
        # Filas de la matriz
        for i in range(num_rows):
            row = []
            cols_row = st.columns(num_cols+1)
            
            # Etiqueta de fila
            with cols_row[0]:
                st.markdown(f"<div style='text-align: right; font-weight: bold; padding-right: 10px;'>{state_names[i]}</div>", 
                          unsafe_allow_html=True)
            
            # Celdas de entrada
            for j in range(num_cols):
                with cols_row[j+1]:
                    default_val = 1.0/num_cols if default_values is None else default_values[i][j]
                    cell_value = st.number_input(
                        f"P({i+1},{j+1})", 
                        min_value=0.0, 
                        max_value=1.0, 
                        value=float(default_val),
                        format=f"%.{decimal_places}f",
                        key=f"{key_prefix}_{i}_{j}",
                        label_visibility="collapsed"
                    )
                    row.append(cell_value)
            matrix.append(row)
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    return matrix

# Sección de entrada de matriz
st.markdown("<h3 style='text-align: center;' class='animate-panel'>Matriz de Transición</h3>", unsafe_allow_html=True)

# Opciones para ingresar la matriz
matrix_input_option = st.radio(
    "Seleccione método para ingresar la matriz",
    ["Manual", "Matriz uniforme", "Matriz aleatoria"],
    horizontal=True
)

transition_matrix = None

if matrix_input_option == "Manual":
    transition_matrix = create_matrix_input(num_states, num_states, "P")
elif matrix_input_option == "Matriz uniforme":
    # Crear matriz con probabilidades uniformes
    uniform_matrix = [[1.0/num_states for _ in range(num_states)] for _ in range(num_states)]
    transition_matrix = create_matrix_input(num_states, num_states, "P_uniform", uniform_matrix)
else:  # Matriz aleatoria
    # Crear matriz aleatoria estocástica
    random_matrix = np.random.rand(num_states, num_states)
    # Normalizar filas para que sumen 1
    random_matrix = random_matrix / random_matrix.sum(axis=1)[:, np.newaxis]
    random_matrix = random_matrix.tolist()
    transition_matrix = create_matrix_input(num_states, num_states, "P_random", random_matrix)

# Normalizar filas para garantizar que sumen 1
normalize = st.checkbox("Normalizar filas automáticamente", value=True)

if normalize:
    # Convertir a numpy para facilitar cálculos
    transition_matrix = np.array(transition_matrix)
    row_sums = transition_matrix.sum(axis=1)
    
    # Evitar división por cero
    row_sums[row_sums == 0] = 1
    
    # Normalizar cada fila
    transition_matrix = transition_matrix / row_sums[:, np.newaxis]

# Verificación visual de que la matriz es estocástica
row_sums = np.sum(transition_matrix, axis=1)
all_rows_sum_to_one = np.allclose(row_sums, np.ones(num_states), rtol=1e-5)

if not all_rows_sum_to_one:
    st.warning("Advertencia: No todas las filas suman 1. Considera activar 'Normalizar filas automáticamente'.")
    st.write("Sumas de filas:", row_sums)
else:
    st.markdown("<div class='success-msg'>✓ La matriz es estocástica (todas las filas suman 1)</div>", unsafe_allow_html=True)

# Creación de la cadena de Markov

    markov_chain = MarkovChain(transition_matrix, states=state_names)
    
    # Detectar estados absorbentes
    absorbing_states = markov_chain.get_absorbing_states()
    if absorbing_states:
        st.markdown(f"<div class='success-msg'>ℹ️ La cadena tiene {len(absorbing_states)} estado(s) absorbente(s): {', '.join([state_names[i] for i in absorbing_states])}</div>", unsafe_allow_html=True)
    
    # Mostrar análisis y opciones adicionales
    st.markdown("<h3 style='text-align: center;' class='animate-panel'>Análisis de la Cadena de Markov</h3>", unsafe_allow_html=True)
    
    # Pestañas para diferentes análisis
    tabs = st.tabs(["Visualización", "Proyección de Estado", "Estado Estacionario", "Análisis de Absorción"])
    
    # Pestaña de visualización
    with tabs[0]:
        st.markdown("<h4>Representación gráfica de la matriz de transición</h4>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Crear heatmap
            fig, ax = plt.subplots(figsize=(7, 5))
            sns.heatmap(
                transition_matrix, 
                annot=True, 
                cmap=selected_colormap, 
                fmt=f'.{decimal_places}f',
                xticklabels=state_names,
                yticklabels=state_names,
                cbar_kws={'label': 'Probabilidad de transición'}
            )
            plt.title("Matriz de Transición")
            plt.tight_layout()
            st.pyplot(fig)
        
        with col2:
            # Grafo de transiciones usando Plotly
            fig = go.Figure()
            
            # Preparar posiciones de los nodos en círculo
            theta = np.linspace(0, 2*np.pi, num_states, endpoint=False)
            radius = 1
            x = radius * np.cos(theta)
            y = radius * np.sin(theta)
            
            # Añadir nodos
            fig.add_trace(go.Scatter(
                x=x, y=y,
                mode='markers+text',
                marker=dict(size=40, color='lightblue', line=dict(width=2, color='darkblue')),
                text=state_names,
                textposition="middle center",
                hoverinfo='text',
                name='Estados'
            ))
            
            # Añadir conexiones/flechas
            for i in range(num_states):
                for j in range(num_states):
                    if transition_matrix[i][j] > 0.01:  # Mostrar solo conexiones significativas
                        # Calcular posición de la curva para las flechas
                        # Auto-transiciones como arcos
                        if i == j:
                            xloop = [x[i], x[i] + 0.2, x[i] + 0.2, x[i]]
                            yloop = [y[i], y[i] + 0.3, y[i] - 0.3, y[i]]
                            
                            fig.add_trace(go.Scatter(
                                x=xloop, y=yloop,
                                mode='lines',
                                line=dict(width=2, color='rgba(70, 130, 180, 0.8)'),
                                hoverinfo='text',
                                text=f"P({state_names[i]} → {state_names[j]}) = {transition_matrix[i][j]:.{decimal_places}f}",
                                showlegend=False
                            ))
                        else:
                            # Control points for Bezier curve
                            # Offset to make curves instead of straight lines
                            midx = (x[i] + x[j]) / 2
                            midy = (y[i] + y[j]) / 2
                            # Perpendicular offset
                            dx = x[j] - x[i]
                            dy = y[j] - y[i]
                            offset = 0.2
                            ctrl_x = midx - dy * offset
                            ctrl_y = midy + dx * offset
                            
                            t = np.linspace(0, 1, 50)
                            # Quadratic Bezier
                            curve_x = (1-t)**2 * x[i] + 2*(1-t)*t * ctrl_x + t**2 * x[j]
                            curve_y = (1-t)**2 * y[i] + 2*(1-t)*t * ctrl_y + t**2 * y[j]
                            
                            fig.add_trace(go.Scatter(
                                x=curve_x, y=curve_y,
                                mode='lines',
                                line=dict(
                                    width=2 + 4 * transition_matrix[i][j],  # Thickness based on probability
                                    color=f'rgba(70, 130, 180, {transition_matrix[i][j]:.2f})'
                                ),
                                hoverinfo='text',
                                text=f"P({state_names[i]} → {state_names[j]}) = {transition_matrix[i][j]:.{decimal_places}f}",
                                showlegend=False
                            ))
                            
                            # Add arrow at the end
                            arrow_size = 10
                            alpha = 0.9  # How far along the curve to place the arrow
                            arrow_idx = int(alpha * (len(curve_x) - 1))
                            end_pt_x = curve_x[arrow_idx]
                            end_pt_y = curve_y[arrow_idx]
                            # Direction vector for arrow
                            try:
                                dx = curve_x[arrow_idx+1] - curve_x[arrow_idx-1]
                                dy = curve_y[arrow_idx+1] - curve_y[arrow_idx-1]
                                norm = np.sqrt(dx**2 + dy**2)
                                if norm > 0:
                                    dx, dy = dx/norm, dy/norm
                            except IndexError:
                                dx, dy = 0, 0
                                
                            # Perpendicular vectors for arrowhead
                            p1x = end_pt_x - arrow_size * (dx*0.866 - dy*0.5)
                            p1y = end_pt_y - arrow_size * (dy*0.866 + dx*0.5)
                            p2x = end_pt_x - arrow_size * (dx*0.866 + dy*0.5)
                            p2y = end_pt_y - arrow_size * (dy*0.866 - dx*0.5)
                            
                            # Draw arrowhead
                            fig.add_trace(go.Scatter(
                                x=[end_pt_x, p1x, p2x, end_pt_x],
                                y=[end_pt_y, p1y, p2y, end_pt_y],
                                fill="toself",
                                fillcolor=f'rgba(70, 130, 180, {transition_matrix[i][j]:.2f})',
                                line=dict(width=0),
                                showlegend=False,
                                hoverinfo='skip'
                            ))
                            
            # Configurar diseño
            fig.update_layout(
                title="Grafo de Transiciones",
                showlegend=False,
                width=600,
                height=500,
                xaxis=dict(
                    showgrid=False,
                    zeroline=False,
                    visible=False,
                    range=[-1.3, 1.3]
                ),
                yaxis=dict(
                    showgrid=False,
                    zeroline=False,
                    visible=False,
                    range=[-1.3, 1.3]
                ),
                margin=dict(l=20, r=20, t=40, b=20),
                plot_bgcolor='rgba(240,240,240,0.8)'
            )
            
            st.plotly_chart(fig)
    
    # Pestaña de proyección de estado
    with tabs[1]:
        st.markdown("<h4>Proyección de estado futuro</h4>", unsafe_allow_html=True)
        
        # Selección de estado inicial
        col1, col2 = st.columns([1, 1])
        
        with col1:
            initial_state_type = st.radio(
                "Tipo de estado inicial",
                ["Estado específico", "Vector de probabilidad personalizado"]
            )
            
            if initial_state_type == "Estado específico":
                initial_state_index = st.selectbox(
                    "Estado inicial",
                    range(num_states),
                    format_func=lambda x: state_names[x]
                )
                initial_state = np.zeros(num_states)
                initial_state[initial_state_index] = 1.0
            else:
                st.write("Definir vector de probabilidad inicial:")
                initial_state = np.zeros(num_states)
                for i in range(num_states):
                    initial_state[i] = st.slider(
                        f"P({state_names[i]})",
                        0.0, 1.0, 1.0/num_states,
                        key=f"init_prob_{i}"
                    )
                
                # Normalizar
                if st.checkbox("Normalizar vector inicial", value=True):
                    initial_state = initial_state / sum(initial_state)
            
            # Número de pasos
            n_steps = st.slider("Número de pasos a proyectar", 1, 50, 10)
            
        with col2:
            # Mostrar distribución de probabilidad actual
            st.markdown("<h5>Estado Inicial</h5>", unsafe_allow_html=True)
            
            df_initial = pd.DataFrame({
                'Estado': state_names,
                'Probabilidad': initial_state
            })
            
            # Gráfico de barras
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=state_names,
                y=initial_state,
                marker_color='lightblue',
                marker_line_color='darkblue',
                marker_line_width=1.5
            ))
            fig.update_layout(
                yaxis=dict(title='Probabilidad', range=[0, 1]),
                title='Distribución de probabilidad inicial',
                plot_bgcolor='white'
            )
            st.plotly_chart(fig)
            
        # Calcular la evolución de estados
        evolution = markov_chain.get_state_evolution(initial_state, n_steps)
        
        # Mostrar resultado
        st.markdown("<h5>Evolución de Estados</h5>", unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Gráfico de evolución
            fig = go.Figure()
            
            for i in range(num_states):
                fig.add_trace(go.Scatter(
                    x=list(range(n_steps+1)),
                    y=evolution[:, i],
                    mode='lines+markers',
                    name=state_names[i]
                ))
            
            fig.update_layout(
                xaxis=dict(title='Paso'),
                yaxis=dict(title='Probabilidad', range=[0, 1]),
                title='Evolución de la distribución de probabilidad',
                legend_title='Estados',
                hovermode='x unified'
            )
            st.plotly_chart(fig)
            
        with col2:
            # Mostrar el resultado final
            st.markdown("<h5>Estado final (después de {} pasos)</h5>".format(n_steps), unsafe_allow_html=True)
            
            df_final = pd.DataFrame({
                'Estado': state_names,
                'Probabilidad': evolution[-1]
            })
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=state_names,
                y=evolution[-1],
                marker_color='lightgreen',
                marker_line_color='darkgreen',
                marker_line_width=1.5
            ))
            fig.update_layout(
                yaxis=dict(title='Probabilidad', range=[0, 1]),
                title='Distribución final',
                plot_bgcolor='white'
            )
            st.plotly_chart(fig)
            
            # Tabla de evolución
            if st.checkbox("Mostrar tabla de evolución"):
                evolution_df = pd.DataFrame(
                    evolution,
                    columns=state_names
                )
                evolution_df.insert(0, 'Paso', range(n_steps+1))
                st.dataframe(evolution_df.style.format({col: f'{{:.{decimal_places}f}}' for col in state_names}))
        
        # Animación de la evolución
        if st.checkbox("Mostrar animación de evolución", value=True):
            st.markdown("<h5>Animación de la evolución</h5>", unsafe_allow_html=True)
            
            # Crear figura para animación
            fig, ax = plt.subplots(figsize=(10, 5))
            
            bar_colors = sns.color_palette(selected_colormap, num_states)
            
            def update_bars(frame):
                ax.clear()
                ax.bar(state_names, evolution[frame], color=bar_colors)
                ax.set_ylim(0, 1)
                ax.set_title(f'Distribución en el paso {frame}')
                ax.set_ylabel('Probabilidad')
                
            anim = FuncAnimation(
                fig, 
                update_bars, 
                frames=n_steps+1,
                interval=500/animation_speed,
                repeat=True
            )
            
            # Guardar animación a un gif en memoria
            buffer = BytesIO()
            import tempfile
            from matplotlib.animation import PillowWriter

# Crear archivo temporal .gif
        with tempfile.NamedTemporaryFile(suffix=".gif", delete=False) as tmpfile:
            anim.save(tmpfile.name, writer='pillow', fps=int(2 * animation_speed))
            tmpfile.seek(0)
            gif_data = base64.b64encode(tmpfile.read()).decode("utf-8")

            
            # Mostrar animación en HTML
            st.markdown(
                f'<img src="data:image/gif;base64,{gif_data}" alt="Animación" width="100%">',
                unsafe_allow_html=True
            )
            
    # Pestaña de estado estacionario
    with tabs[2]:
        st.markdown("<h4>Análisis del Estado Estacionario</h4>", unsafe_allow_html=True)
        
        # Calcular estado estacionario
        pi = markov_chain.steady_state()
        
        if pi is not None and not markov_chain.is_absorbing():
            st.markdown("<div class='success-msg'>✓ La cadena tiene una única distribución estacionaria</div>", unsafe_allow_html=True)
            
            # Mostrar distribución estacionaria
            st.markdown("<h5>Distribución Estacionaria (π)</h5>", unsafe_allow_html=True)
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                pi_df = pd.DataFrame({
                    'Estado': state_names,
                    'Probabilidad': pi
                })
                st.dataframe(pi_df.style.format({'Probabilidad': f'{{:.{decimal_places}f}}'}))
            
            with col2:
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=state_names,
                    y=pi,
                    marker_color='lightblue',
                    marker_line_color='darkblue',
                    marker_line_width=1.5
                ))
                fig.update_layout(
                    yaxis=dict(title='Probabilidad', range=[0, 1]),
                    title='Distribución Estacionaria',
                    plot_bgcolor='white'
                )
                st.plotly_chart(fig)
                
            # Verificación
            st.markdown("<h5>Verificación de π = π⋅P</h5>", unsafe_allow_html=True)
            
            pi_P = pi @ transition_matrix
            
            verification_df = pd.DataFrame({
                'Estado': state_names,
                'π': pi,
                'π⋅P': pi_P,
                'Diferencia': pi - pi_P
            })
            
            st.dataframe(verification_df.style.format({col: f'{{:.{decimal_places}f}}' for col in ['π', 'π⋅P', 'Diferencia']}))
            
            # Demostración de convergencia
            if st.checkbox("Mostrar demostración de convergencia"):
                st.markdown("<h5>Demostración de Convergencia</h5>", unsafe_allow_html=True)
                
                # Seleccionar estado inicial aleatorio
                p0 = np.random.rand(num_states)
                p0 = p0 / np.sum(p0)
                
                # Calcular evolución
                steps = 50
                convergence = np.zeros((steps+1, num_states))
                convergence[0] = p0
                
                for i in range(1, steps+1):
                    convergence[i] = convergence[i-1] @ transition_matrix
                
                # Graficar convergencia
                fig = go.Figure()
                
                for i in range(num_states):
                    fig.add_trace(go.Scatter(
                        x=list(range(steps+1)),
                        y=convergence[:, i],
                        mode='lines',
                        name=state_names[i]
                    ))
                
                # Añadir líneas horizontales para la distribución estacionaria
                for i in range(num_states):
                    fig.add_trace(go.Scatter(
                        x=[0, steps],
                        y=[pi[i], pi[i]],
                        mode='lines',
                        line=dict(dash='dash', color='black', width=1),
                        showlegend=False
                    ))
                
                fig.update_layout(
                    xaxis=dict(title='Paso'),
                    yaxis=dict(title='Probabilidad', range=[0, 1]),
                    title='Convergencia a la distribución estacionaria',
                    legend_title='Estados'
                )
                st.plotly_chart(fig)
                
                # Mostrar distancia a la distribución estacionaria
                distances = np.sum(np.abs(convergence - pi), axis=1)
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=list(range(steps+1)),
                    y=distances,
                    mode='lines+markers'
                ))
                fig.update_layout(
                    xaxis=dict(title='Paso'),
                    yaxis=dict(title='Distancia L1 a π', type='log'),
                    title='Distancia a la distribución estacionaria (escala logarítmica)'
                )
                st.plotly_chart(fig)
        
        elif markov_chain.is_absorbing():
            st.warning("Esta cadena contiene estados absorbentes, por lo que no tiene una única distribución estacionaria.")
            st.markdown("Para cadenas con estados absorbentes, consulta la pestaña 'Análisis de Absorción'.")
        else:
            st.warning("Esta cadena no parece tener una única distribución estacionaria.")
            st.markdown("Posibles razones:")
            st.markdown("- La cadena puede tener múltiples componentes periódicos")
            st.markdown("- La cadena puede no ser irreducible (algunos estados no son alcanzables desde otros)")
    
    # Pestaña de análisis de absorción
    with tabs[3]:
        st.markdown("<h4>Análisis de Estados Absorbentes</h4>", unsafe_allow_html=True)
        
        absorbing_states = markov_chain.get_absorbing_states()
        
        if absorbing_states:
            st.markdown("<div class='success-msg'>✓ La cadena tiene estados absorbentes</div>", unsafe_allow_html=True)
            
            # Forma canónica
            canonical_form = markov_chain.canonical_form()
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown("<h5>Estados Absorbentes:</h5>", unsafe_allow_html=True)
                for idx in absorbing_states:
                    st.markdown(f"- {state_names[idx]}")
                
                st.markdown("<h5>Estados No Absorbentes:</h5>", unsafe_allow_html=True)
                non_absorbing = canonical_form["non_absorbing"]
                for idx in non_absorbing:
                    st.markdown(f"- {state_names[idx]}")
            
            with col2:
                st.markdown("<h5>Forma Canónica de la Matriz:</h5>", unsafe_allow_html=True)
                
                # Crear y mostrar la matriz en forma canónica
                fig, ax = plt.subplots(figsize=(7, 5))
                
                # Dibujar líneas divisorias
                t = len(absorbing_states)
                r = len(non_absorbing)
                
                sns.heatmap(
                    canonical_form["P_canonical"], 
                    annot=True, 
                    cmap=selected_colormap, 
                    fmt=f'.{decimal_places}f',
                    xticklabels=[state_names[i] for i in canonical_form["order"]],
                    yticklabels=[state_names[i] for i in canonical_form["order"]],
                    cbar_kws={'label': 'Probabilidad'}
                )
                
                # Añadir líneas para separar bloques I, 0, R, Q
                plt.axhline(y=t, color='r', linestyle='-')
                plt.axvline(x=t, color='r', linestyle='-')
                
                # Añadir etiquetas para cada bloque
                plt.text(t/2, t/2, "I", color='white', fontsize=20, ha='center', va='center')
                plt.text(t + r/2, t/2, "0", color='white', fontsize=20, ha='center', va='center')
                plt.text(t/2, t + r/2, "R", color='white', fontsize=20, ha='center', va='center')
                plt.text(t + r/2, t + r/2, "Q", color='white', fontsize=20, ha='center', va='center')
                
                plt.title("Matriz de Transición en Forma Canónica")
                plt.tight_layout()
                st.pyplot(fig)
            
            # Calcular probabilidades de absorción
            absorption_probs = markov_chain.absorption_probabilities()
            
            if absorption_probs is not None:
                st.markdown("<h5>Probabilidades de Absorción</h5>", unsafe_allow_html=True)
                
                # Crear dataframe para probabilidades de absorción
                absorbing_names = [state_names[i] for i in absorbing_states]
                non_absorbing_names = [state_names[i] for i in canonical_form["non_absorbing"]]
                
                absorption_df = pd.DataFrame(
                    absorption_probs,
                    columns=absorbing_names,
                    index=non_absorbing_names
                )
                
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.markdown("*Filas: Estados iniciales (no absorbentes)*")
                    st.markdown("*Columnas: Estados finales (absorbentes)*")
                    st.dataframe(absorption_df.style.format({col: f'{{:.{decimal_places}f}}' for col in absorbing_names}))
                
                with col2:
                    # Visualización de probabilidades de absorción
                    fig = go.Figure()
                    
                    for i, absorbing in enumerate(absorbing_names):
                        fig.add_trace(go.Bar(
                            x=non_absorbing_names,
                            y=absorption_probs[:, i],
                            name=f'Absorción en {absorbing}'
                        ))
                    
                    fig.update_layout(
                        barmode='stack',
                        xaxis=dict(title='Estado Inicial'),
                        yaxis=dict(title='Probabilidad de Absorción', range=[0, 1]),
                        title='Probabilidades de Absorción por Estado Inicial',
                        legend_title='Estado Absorbente'
                    )
                    st.plotly_chart(fig)
                
                # Tiempo esperado hasta la absorción
                expected_steps = markov_chain.expected_steps_to_absorption()
                
                if expected_steps is not None:
                    st.markdown("<h5>Tiempo Esperado hasta la Absorción</h5>", unsafe_allow_html=True)
                    
                    col1, col2 = st.columns([1, 1])
                    
                    with col1:
                        steps_df = pd.DataFrame({
                            'Estado Inicial': non_absorbing_names,
                            'Pasos Esperados': expected_steps
                        })
                        st.dataframe(steps_df.style.format({'Pasos Esperados': f'{{:.{decimal_places}f}}'}))
                    
                    with col2:
                        fig = go.Figure()
                        fig.add_trace(go.Bar(
                            x=non_absorbing_names,
                            y=expected_steps,
                            marker_color='lightgreen',
                            marker_line_color='darkgreen',
                            marker_line_width=1.5
                        ))
                        fig.update_layout(
                            xaxis=dict(title='Estado Inicial'),
                            yaxis=dict(title='Pasos Esperados'),
                            title='Pasos Esperados hasta la Absorción',
                            plot_bgcolor='white'
                        )
                        st.plotly_chart(fig)
        else:
            st.info("Esta cadena no tiene estados absorbentes.")
            st.markdown("Un estado absorbente es aquel que, una vez alcanzado, no puede ser abandonado (probabilidad de transición a sí mismo = 1).")
            st.markdown("Para analizar el comportamiento a largo plazo de esta cadena, consulta la pestaña 'Estado Estacionario'.")

# Añadir una sección de documentación e información
with st.expander("Documentación y Ayuda"):
    st.markdown("""
    ## Acerca de las Cadenas de Markov
    
    Una **cadena de Markov** es un proceso estocástico que satisface la propiedad de Markov: la probabilidad de pasar a un estado futuro depende únicamente del estado actual, no de la secuencia de eventos que lo precedieron.
    
    ### Conceptos clave:
    
    1. **Matriz de transición (P)**: Matriz donde P(i,j) representa la probabilidad de pasar del estado i al estado j en un solo paso.
    
    2. **Estado absorbente**: Un estado que, una vez alcanzado, no puede ser abandonado (la probabilidad de transición a sí mismo es 1).
    
    3. **Distribución estacionaria**: Vector π tal que π = π·P, es decir, permanece invariante bajo la multiplicación por la matriz de transición.
    
    4. **Forma canónica**: Para cadenas con estados absorbentes, reorganización de la matriz en la forma:
       
       ```
       P = [ I  0 ]
           [ R  Q ]
       ```
       
       donde I es una matriz identidad para estados absorbentes, 0 es una matriz de ceros, R contiene probabilidades de transición de estados no absorbentes a absorbentes, y Q contiene probabilidades entre estados no absorbentes.
    
    ### Cálculos importantes:
    
    - **Probabilidades de absorción**: B = N·R, donde N = (I-Q)^(-1) es la matriz fundamental.
    - **Pasos esperados hasta la absorción**: t = N·1, donde 1 es un vector de unos.
    
    ### Cómo usar esta calculadora:
    
    1. Define el número de estados y sus nombres (opcional).
    2. Ingresa la matriz de transición manualmente o utiliza las opciones predefinidas.
    3. Explora las diferentes pestañas para analizar:
       - Visualización de la cadena
       - Proyección de estados futuros
       - Análisis del estado estacionario
       - Propiedades de absorción (si aplica)
    """)

# Footer
st.markdown("""
<div style="text-align: center; margin-top: 30px; padding-top: 20px; border-top: 1px solid #eee;">
    <p>Calculadora de Cadenas de Markov | Desarrollada con Streamlit y Python</p>
</div>
""", unsafe_allow_html=True)