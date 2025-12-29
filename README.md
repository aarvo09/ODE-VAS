# ODE VAS

**ODE VAS** (Ordinary Differential Equation Visualization and Analysis System) is a powerful, modern web-based platform for solving, visualizing, and analyzing first-order ordinary differential equations. It combines analytical methods with numerical techniques to provide comprehensive insights into ODE behavior, stability, and convergence.

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

### **Example Equations**
`-2*x*y` (decay), `y*(1-y)` (logistic), `-x/y` (oscillator), `sin(x)*y`, `exp(-x)*y`  
**Supported**: `sin`, `cos`, `tan`, `exp`, `log`, `sqrt`, `+`, `-`, `*`, `/`, `**`

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
    git clone <repository-url>
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

## Project Structure

```
ODE VAS/
├── app.py                   # Flask backend & ODE solvers
├── requirements.txt         # Dependencies
├── templates/               # HTML pages (home, quick-solver, visualizer, advanced)
└── static/                  # CSS styles & JavaScript (utils, visualizer, advanced)
```

---

## Acknowledgments

*   **Flask** - Python web framework
*   **SymPy** - Symbolic computation library
*   **Chart.js** - Beautiful JavaScript charts
*   **NumPy** - Numerical computing

---


## 📝 License

This project is open source and available under the MIT License.

---

## 👨‍💻 Author

**Arvind**
- GitHub: [Your GitHub Profile]
- Email: [Your Email]

---

