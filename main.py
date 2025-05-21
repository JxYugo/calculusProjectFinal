import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from sympy import symbols, lambdify, sympify, diff, integrate
import tkinter as tk
from tkinter import messagebox, ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Numerical derivative
def numerical_derivative(fx, x_vals, order=1, h=1e-5):
    if order == 1:
        return (fx(x_vals + h) - fx(x_vals - h)) / (2 * h)
    elif order == 2:
        return (fx(x_vals + h) - 2 * fx(x_vals) + fx(x_vals - h)) / (h ** 2)
    else:
        raise ValueError("Only 1st and 2nd derivatives are supported.")

# Numerical integration
def numerical_integral(fx, x_vals):
    x_start = x_vals[0]
    result = [0]
    for x in x_vals[1:]:
        val, _ = quad(fx, x_start, x)
        result.append(val)
    return np.array(result)

# GUI Application
class CalculusVisualizer:
    def __init__(self, root):
        self.root = root
        self.root.title("Calculus Visualizer")

        # Input fields
        tk.Label(root, text="Function f(x):").grid(row=0, column=0, sticky="e")
        self.func_entry = tk.Entry(root, width=30)
        self.func_entry.grid(row=0, column=1, columnspan=2, pady=5)

        tk.Label(root, text="x-min:").grid(row=1, column=0, sticky="e")
        self.xmin_entry = tk.Entry(root, width=10)
        self.xmin_entry.grid(row=1, column=1)

        tk.Label(root, text="x-max:").grid(row=1, column=2, sticky="e")
        self.xmax_entry = tk.Entry(root, width=10)
        self.xmax_entry.grid(row=1, column=3)

        tk.Label(root, text="Derivative Order:").grid(row=2, column=0, sticky="e")
        self.derivative_order = ttk.Combobox(root, values=[1, 2], width=5)
        self.derivative_order.current(0)
        self.derivative_order.grid(row=2, column=1, pady=5)

        # Buttons
        tk.Button(root, text="Plot", command=self.plot).grid(row=2, column=2)
        tk.Button(root, text="Save Plot", command=self.save_plot).grid(row=2, column=3)

        # Plot area
        self.fig, self.ax = plt.subplots(figsize=(8, 5))
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().grid(row=3, column=0, columnspan=4)

        # Labels to show function, derivative, and integral
        self.func_label = tk.Label(root, text="Function: ", fg="blue", anchor="w", font=("Arial", 10, "bold"), wraplength=600)
        self.func_label.grid(row=4, column=0, columnspan=4, sticky="w", padx=10)

        self.deriv_label = tk.Label(root, text="Derivative: ", fg="red", anchor="w", font=("Arial", 10, "bold"), wraplength=600)
        self.deriv_label.grid(row=5, column=0, columnspan=4, sticky="w", padx=10)

        self.integral_label = tk.Label(root, text="Integral: ", fg="green", anchor="w", font=("Arial", 10, "bold"), wraplength=600)
        self.integral_label.grid(row=6, column=0, columnspan=4, sticky="w", padx=10)

    def plot(self):
        x = symbols('x')
        func_str = self.func_entry.get()

        try:
            expr = sympify(func_str)
            fx = lambdify(x, expr, modules=['numpy'])
        except Exception as e:
            messagebox.showerror("Error", f"Invalid function: {e}")
            return

        try:
            xmin = float(self.xmin_entry.get())
            xmax = float(self.xmax_entry.get())
            if xmin == xmax:
                raise ValueError("xmin and xmax cannot be equal.")
            x_vals = np.linspace(xmin, xmax, 400)
        except Exception as e:
            messagebox.showerror("Error", f"Invalid x-range: {e}")
            return

        try:
            order = int(self.derivative_order.get())
            y_vals = fx(x_vals)
            dy_vals = numerical_derivative(fx, x_vals, order=order)
            integral_vals = numerical_integral(fx, x_vals)

            # Symbolic computation
            deriv_expr = diff(expr, x, order)
            integral_expr = integrate(expr, x)

            # Display symbolic expressions
            self.func_label.config(text=f"Function: f(x) = {str(expr)}")
            self.deriv_label.config(text=f"Derivative: f'(x) = {str(deriv_expr)}")
            self.integral_label.config(text=f"Integral: ∫f(x)dx = {str(integral_expr)}")

        except Exception as e:
            messagebox.showerror("Error", f"Computation failed: {e}")
            return

        self.ax.clear()
        self.ax.plot(x_vals, y_vals, label='Original Function', color='blue')
        self.ax.plot(x_vals, dy_vals, label=f'{order}st Derivative' if order == 1 else '2nd Derivative', linestyle='--', color='red')
        self.ax.plot(x_vals, integral_vals, label='Integral', linestyle='-.', color='green')
        self.ax.set_title("Function, Derivative, and Integral")
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")
        self.ax.legend()
        self.ax.grid(True)
        self.fig.tight_layout()
        self.canvas.draw()

    def save_plot(self):
        try:
            self.fig.savefig("calculus_plot_gui.png")
            messagebox.showinfo("Saved", "Plot saved as 'calculus_plot_gui.png'")
        except Exception as e:
            messagebox.showerror("Save Error", f"Could not save plot: {e}")

# Launch GUI
if __name__ == "__main__":
    root = tk.Tk()
    app = CalculusVisualizer(root)
    root.mainloop()
