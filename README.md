# ODE VAS

**ODE VAS** (Ordinary Differential Equation Visualization and Analysis System) is a powerful, modern web-based platform for solving, visualizing, and analyzing first-order ordinary differential equations. It combines analytical methods with numerical techniques to provide comprehensive insights into ODE behavior, stability, and convergence.

## About

- Try it here : [link](https://ode-vas.onrender.com/)

## Demo 

- Demo video of the project : [Video](https://drive.google.com/file/d/1fLSao6DwAKwZjGYRYRQhhWRT-JbWvAK7/view?usp=drive_link)

## Features & Usage
    
### **Quick Solver**
Fast numerical solutions with tabular output. Select a method (Euler, RK4, or analytical methods like Separation of Variables), enter your equation, set initial conditions, and solve.

### **Interactive Visualizer**
Real-time graphing with dynamic controls:
*   Visualize solutions with smooth curve plotting
*   Enable **slope fields** (direction fields) with customizable density, color, and opacity
*   Use **interactive sliders** to adjust y₀ and x_end with instant graph updates
*   View **live statistics**: max/min values, final points, and solution behavior

### **Advanced Analysis**
Comprehensive ODE analysis with multiple tools:
*   **Parameter Variation**: Compare solutions across parameter ranges with color-coded multi-line graphs
*   **Step-Size Comparison**: Analyze convergence rates and error metrics across different step sizes
*   **Stability Analysis**: Automatic equilibrium point detection with stability classification (stable/unstable/neutral)
*   **Phase Portrait**: Generate flow fields with 7 trajectories from different initial conditions

### **Solving Methods**
*   **Numerical**: Euler, Improved Euler, Runge-Kutta 4th Order (RK4)
*   **Analytical**: Direct Integration, Separation of Variables, Integrating Factor, Substitution

---

##  Example Equations

*   **Exponential Decay**: `-2*x*y`
*   **Logistic Growth**: `y*(1-y)`
*   **Harmonic Oscillator**: `-x/y`
*   **Polynomial**: `x**2 - y`
*   **Trigonometric**: `sin(x)*y`
*   **Exponential Function**: `exp(-x)*y`

### Supported Functions
`sin`, `cos`, `tan`, `exp`, `log`, `sqrt`, `+`, `-`, `*`, `/`, `**` (power)

---

##  Features in Detail

### Parameter Variation
Compare solutions across different parameter values (e.g., varying decay rates or growth constants). Results displayed as color-coded multi-line graphs with HSL gradient coloring.

### Step-Size Comparison
Analyze numerical accuracy by comparing solutions with different step sizes. View:
- Number of evaluation points
- Maximum and mean errors (compared to reference)
- Convergence rates between consecutive step sizes

### Stability Analysis
Automatically finds equilibrium points by solving dy/dx = 0. Classifies stability based on derivative:
- **Stable**: f'(y) < 0 (solutions converge)
- **Unstable**: f'(y) > 0 (solutions diverge)
- **Neutral**: f'(y) = 0 (center/saddle point)

### Phase Portrait
Generates 7 trajectories from different starting points across the domain. Displays equilibrium lines as dashed horizontals with color-coded stability.

---

##  UI Features

*   **Real-Time Validation**: Input fields validate as you type with visual feedback
*   **Glassmorphism Design**: Modern blur effects on panels and controls
*   **Toast Notifications**: Non-intrusive feedback for actions
*   **Responsive Layout**: Works on desktop, tablet, and mobile
*   **Dark Theme**: Easy on the eyes with red accent colors
*   **Keyboard Shortcuts**: ESC to close modals
*   **Form Persistence**: Restore previous analysis with one click

---

---

##  Technology Stack

### **Backend (Python)**
*   **Flask** - Web framework
*   **SymPy** - Symbolic mathematics and analytical solving
*   **NumPy** - Numerical computations

### **Frontend**
*   **HTML5 / CSS3** - Modern responsive UI with glassmorphism effects
*   **JavaScript (ES6)** - Interactive controls and real-time updates
*   **Chart.js** - High-performance data visualization

### **Design**
*   Red/Black theme with glassmorphism
*   Smooth transitions and animations
*   Mobile-responsive layout

---

## Installation & Setup

1.  **Prerequisites**:
    *   Python 3.8 or higher
    *   Modern Web Browser (Chrome, Firefox, Edge, Safari)

2.  **Clone the Repository**:
    ```bash
    git clone https://github.com/aarvo09/ODE-VAS
    cd "ODE VAS"
    ```

3.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the Application**:
    ```bash
    python app.py
    ```

5.  **Access the Application**:
    *   Open your browser and navigate to: `http://localhost:5000`

---

##  Project Structure

```
ODE VAS/
├── .venv/                     
├── static/
│   ├── images/
│   │   └── math_doodles.png
│   ├── js/
│   │   ├── advanced.js
│   │   ├── quick-solver.js
│   │   ├── utils.js
│   │   └── visualizer.js
│   ├── advanced.css
│   ├── home.css
│   ├── quick-solver.css
│   ├── style.css
│   ├── transitions.css
│   └── visualizer.css
├── templates/
│   ├── advanced.html
│   ├── help.html
│   ├── home.html
│   ├── quick-solver.html
│   ├── solver.html
├── .gitignore
├── app.py
├── LICENSE
├── README.md
└── requirements.txt                  
```    

```

##  License

This project is open source and available under the MIT License.

---

##  Author

**Arvind**
- GitHub: [https://github.com/aarvo09]
- Email: [arvindkumarsingh1008n@gmail.com]

---
