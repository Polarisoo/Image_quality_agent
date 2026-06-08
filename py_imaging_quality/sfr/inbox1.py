"""SFR parameter input dialog, equivalent to MATLAB inbox1.m."""
import tkinter as tk
from tkinter import simpledialog


def inbox1():
    """Get SFR calculation parameters from user.

    Returns:
        num: number of ROIs
        gamma: gamma value
        weight: RGB to luminance weights [R_w, G_w, B_w]
    """
    # Default values
    num = 1
    gamma = 0.5
    weight = [0.213, 0.715, 0.072]
    return num, gamma, weight
