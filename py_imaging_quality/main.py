"""Main GUI entry point, equivalent to MATLAB main.m.

Pure matplotlib GUI — all widgets (buttons, image display, ROI selection)
run in a single matplotlib event loop, avoiding tkinter conflicts.
"""
import os
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, RectangleSelector
from matplotlib.patches import Rectangle

from PIL import Image
from tkinter import filedialog
import tkinter as tk


class ImagingQualityApp:
    """Imaging quality test application using pure matplotlib GUI.

    Workflow:
      1. Click "Open Image" → file dialog → image displayed
      2. Click test button (e.g. "SFR") → status updates
      3. Draw ROI rectangle(s) directly on the image
      4. Click "Compute" → results shown in popup figure
    """

    def __init__(self):
        self.img = None
        self.k1 = 0.0
        self.img_corrected = None

        # ROI state
        self.rois = []           # list of ROI image arrays
        self.roi_coords = []     # list of (ul_x, ul_y, lr_x, lr_y)
        self.roi_patches = []    # matplotlib Rectangle patches on axes
        self.current_mode = None
        self.roi_count_needed = 0

        # Build the figure
        self.fig = plt.figure(figsize=(12, 8))
        self.fig.canvas.manager.set_window_title('Image Quality Test')

        # Buttons in left margin, image in main area
        self._build_buttons()

        self.ax_img = self.fig.add_axes([0.22, 0.08, 0.76, 0.88])
        self.ax_img.set_xticks([])
        self.ax_img.set_yticks([])

        # Status bar
        self.fig.text(0.5, 0.01, 'Welcome! Open an image to begin.',
                      ha='center', va='center', fontsize=9, color='gray')

        # RectangleSelector — created once, reused
        self._selector = None
        self._create_selector()

    # ------------------------------------------------------------------
    #  Button grid
    # ------------------------------------------------------------------
    def _build_buttons(self):
        bw, bh = 0.18, 0.06  # button width, height
        bx = 0.02            # left edge for all buttons
        by_start = 0.88      # topmost button bottom
        by_gap = 0.09        # vertical gap between buttons

        buttons = [
            ("Open Image",     self._on_open,         (bx, by_start, bw, bh), '#ddd'),
            ("SFR (1 ROI)",    lambda e: self._start_mode('sfr', 1),
                                                     (bx, by_start - by_gap, bw, bh), '#aad'),
            ("Dist. Meas.",    lambda e: self._start_mode('dist', 1),
                                                     (bx, by_start - 2*by_gap, bw, bh), '#aad'),
            ("Dist. Correct",  self._on_dist_correct, (bx, by_start - 3*by_gap, bw, bh), '#ada'),
            ("Chessboard Cal.",self._on_chess_cal,    (bx, by_start - 4*by_gap, bw, bh), '#ada'),
            ("ColorChk (24)",  lambda e: self._start_mode('color', 24),
                                                     (bx, by_start - 5*by_gap, bw, bh), '#daa'),
            ("StepChrt (20)",  lambda e: self._start_mode('step', 20),
                                                     (bx, by_start - 6*by_gap, bw, bh), '#daa'),
            ("WB (6 patches)", lambda e: self._start_mode('wb', 6),
                                                     (bx, by_start - 7*by_gap, bw, bh), '#daa'),
            ("Dark Field",     self._on_dark_field,   (bx, by_start - 8*by_gap, bw, bh), '#ccc'),
        ]

        self._btns = {}
        for label, cb, rect, color in buttons:
            if label == "":
                continue
            ax_btn = self.fig.add_axes(rect)
            btn = Button(ax_btn, label, color=color, hovercolor='#fff')
            if cb is not None:
                btn.on_clicked(cb)
            self._btns[label] = btn

        # Bottom action bar: Undo + Compute
        ax_undo = self.fig.add_axes((bx, 0.03, 0.09, 0.05))
        self._btn_undo = Button(ax_undo, 'Undo ROI', color='#fdb', hovercolor='#fff')
        self._btn_undo.on_clicked(self._undo_roi)

        ax_compute = self.fig.add_axes((bx + 0.095, 0.03, 0.085, 0.05))
        self._btn_compute = Button(ax_compute, 'Compute', color='#5b9', hovercolor='#fff')
        self._btn_compute.on_clicked(self._compute)

    # ------------------------------------------------------------------
    #  RectangleSelector
    # ------------------------------------------------------------------
    def _create_selector(self):
        if self._selector is not None:
            try:
                self._selector.set_active(False)
            except Exception:
                pass
            self._selector = None

        self._selector = RectangleSelector(
            self.ax_img,
            self._on_roi,
            useblit=False,
            button=[1],
            minspanx=5,
            minspany=5,
            spancoords='pixels',
            interactive=True,
        )

    # ------------------------------------------------------------------
    #  ROI drawing callback — called when user releases mouse
    # ------------------------------------------------------------------
    def _on_roi(self, eclick, erelease):
        if self.img is None:
            return
        if self.current_mode is None:
            self._status('Select a test mode first (e.g. SFR).')
            return
        if len(self.rois) >= self.roi_count_needed:
            self._status(f'Already have {self.roi_count_needed} ROI(s). Press Compute or Undo.')
            return
        if eclick.xdata is None or erelease.xdata is None:
            return

        x1, y1 = int(eclick.xdata), int(eclick.ydata)
        x2, y2 = int(erelease.xdata), int(erelease.ydata)
        ul_x, lr_x = sorted([x1, x2])
        ul_y, lr_y = sorted([y1, y2])

        h, w = self.img.shape[:2]
        ul_x = max(0, ul_x); ul_y = max(0, ul_y)
        lr_x = min(w - 1, lr_x); lr_y = min(h - 1, lr_y)

        roi = self.img[ul_y:lr_y + 1, ul_x:lr_x + 1].copy()
        self.rois.append(roi)
        self.roi_coords.append((ul_x, ul_y, lr_x, lr_y))

        rect = Rectangle((ul_x, ul_y), lr_x - ul_x, lr_y - ul_y,
                         linewidth=2, edgecolor='lime', facecolor='none')
        self.ax_img.add_patch(rect)
        self.roi_patches.append(rect)
        self.fig.canvas.draw()

        n_done = len(self.rois)
        if n_done >= self.roi_count_needed:
            self._status(f'{n_done}/{self.roi_count_needed} ROIs done! Press "Compute".')
        else:
            self._status(f'ROI {n_done}/{self.roi_count_needed} drawn. Draw more or press Compute.')

    def _undo_roi(self, event=None):
        if not self.roi_patches:
            return
        self.roi_patches[-1].remove()
        self.roi_patches.pop()
        self.rois.pop()
        self.roi_coords.pop()
        self.fig.canvas.draw()
        n = len(self.rois)
        self._status(f'Undo. {n}/{self.roi_count_needed} ROIs.')

    def _clear_rois(self):
        for p in self.roi_patches:
            p.remove()
        self.roi_patches.clear()
        self.rois.clear()
        self.roi_coords.clear()

    # ------------------------------------------------------------------
    #  Status text
    # ------------------------------------------------------------------
    def _status(self, text):
        for txt in self.fig.texts:
            try:
                txt.remove()
            except Exception:
                pass
        self.fig.text(0.5, 0.01, text, ha='center', va='center',
                      fontsize=9, color='gray')
        self.fig.canvas.draw()

    # ------------------------------------------------------------------
    #  File open
    # ------------------------------------------------------------------
    def _on_open(self, event=None):
        root = tk.Tk()
        root.withdraw()
        path = filedialog.askopenfilename(
            title='Open image',
            filetypes=[('Images', '*.jpg;*.jpeg;*.png;*.bmp;*.tif;*.tiff'),
                       ('All files', '*.*')])
        root.destroy()
        if not path:
            return
        try:
            self.img = np.array(Image.open(path))
        except Exception as e:
            self._status(f'Error: {e}')
            return
        self._clear_rois()
        self.current_mode = None
        self.ax_img.clear()
        self.ax_img.set_xticks([]); self.ax_img.set_yticks([])
        if self.img.ndim == 2 or self.img.shape[2] == 1:
            self.ax_img.imshow(self.img.squeeze(), cmap='gray')
        else:
            self.ax_img.imshow(self.img)
        self.fig.canvas.draw()
        self._create_selector()
        fname = os.path.basename(path)
        self._status(f'Loaded: {fname}  ({self.img.shape[1]}x{self.img.shape[0]})')

    # ------------------------------------------------------------------
    #  Mode start
    # ------------------------------------------------------------------
    def _start_mode(self, mode, count):
        if self.img is None:
            self._on_open()
            if self.img is None:
                return
        self._clear_rois()
        self.current_mode = mode
        self.roi_count_needed = count
        mode_names = {'sfr': 'SFR', 'dist': 'Distortion', 'color': 'ColorCheck',
                      'step': 'Step Chart', 'wb': 'White Balance'}
        name = mode_names.get(mode, mode)
        self._status(f'Mode: {name}. Draw {count} ROI(s) on the image.')

    # ------------------------------------------------------------------
    #  Compute
    # ------------------------------------------------------------------
    def _compute(self, event=None):
        if self.img is None:
            self._status('No image loaded.')
            return
        if self.current_mode is None:
            self._status('Select a test mode first (e.g. click SFR).')
            return
        if len(self.rois) < self.roi_count_needed:
            self._status(f'Need {self.roi_count_needed} ROI(s), have {len(self.rois)}. Draw more.')
            return

        try:
            if self.current_mode == 'sfr':
                self._compute_sfr()
            elif self.current_mode == 'dist':
                self._compute_distortion()
            elif self.current_mode == 'color':
                self._compute_color()
            elif self.current_mode == 'step':
                self._compute_step()
            elif self.current_mode == 'wb':
                self._compute_wb()
        except Exception as e:
            import traceback
            traceback.print_exc()
            self._status(f'Error: {e}')

    # --- SFR ---
    def _compute_sfr(self):
        from .sfr.sfr_cal import _sfr_cal_from_roi
        from .sfr.mtf_eff import mtf_eff
        freq, mtf, esf = _sfr_cal_from_roi(self.rois, gamma=0.5,
                                           weight=[0.213, 0.715, 0.072])
        if freq is None:
            self._status('SFR returned no results.')
            return

        for num, (f, m, e) in enumerate(zip(freq, mtf, esf)):
            cof = mtf_eff(f, m, [0.5, 0.3])
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7), num=f'SFR ROI {num+1}')
            fig.suptitle(f'SFR / MTF  —  ROI {num + 1}')
            ax1.set_title('Edge Spread Function'); ax1.grid(True)
            if e.shape[1] >= 4:
                ax1.plot(e[:, 0], '--r', lw=1, label='R')
                ax1.plot(e[:, 1], '-g', lw=1, label='G')
                ax1.plot(e[:, 2], '-.b', lw=1, label='B')
                ax1.plot(e[:, 3], '-k', lw=2, label='Lum')
            else:
                ax1.plot(e[:, 0], '-k', lw=2, label='Lum')
            ax1.set_xlabel('Pixel'); ax1.set_ylabel('Value'); ax1.legend(fontsize=8)
            ax2.set_title('MTF Curve'); ax2.grid(True)
            if m.shape[1] >= 4:
                ax2.plot(f, m[:, 0], '--r', lw=1,
                         label=f'R  MTF50={cof[0,0]:.4f}  MTF30={cof[1,0]:.4f}')
                ax2.plot(f, m[:, 1], '-g', lw=1,
                         label=f'G  MTF50={cof[0,1]:.4f}  MTF30={cof[1,1]:.4f}')
                ax2.plot(f, m[:, 2], '-.b', lw=1,
                         label=f'B  MTF50={cof[0,2]:.4f}  MTF30={cof[1,2]:.4f}')
                ax2.plot(f, m[:, 3], '-k', lw=2,
                         label=f'L  MTF50={cof[0,3]:.4f}  MTF30={cof[1,3]:.4f}')
            else:
                ax2.plot(f, m[:, 0], '-k', lw=2,
                         label=f'L  MTF50={cof[0,0]:.4f}  MTF30={cof[1,0]:.4f}')
            ax2.set_xlabel('Frequency (cy/pixel)'); ax2.set_ylabel('MTF')
            ax2.set_xlim(0, f[int(len(f) * 0.75)]); ax2.legend(fontsize=8)
            fig.tight_layout()
            plt.show(block=False)
        self._status('SFR done. See popup figures for MTF50/MTF30.')

    # --- Distortion ---
    def _compute_distortion(self):
        from .distortion.distortion import _distortion_from_roi
        k1, TV = _distortion_from_roi(self.rois[0])
        self.k1 = k1
        self._status(f'k1={k1:.6f}  SMIA TV={100*TV:.2f}%')

    def _on_dist_correct(self, event=None):
        if self.img is None:
            self._on_open()
            if self.img is None:
                return
        from .distortion.distortion_correct import _distortion_correct_from_img
        import tkinter.simpledialog as sd
        root = tk.Tk(); root.withdraw()
        k = sd.askfloat('Distortion Correction', 'k1 value:',
                        initialvalue=self.k1 if self.k1 else 0.1)
        root.destroy()
        if k is None:
            return
        self.k1 = k
        corrected = _distortion_correct_from_img(k, self.img)
        if corrected is not None:
            self.img_corrected = corrected
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
            ax1.imshow(self.img, cmap='gray' if self.img.ndim == 2 else None)
            ax1.set_title('Original'); ax1.axis('off')
            ax2.imshow(corrected, cmap='gray' if corrected.ndim == 2 else None)
            ax2.set_title('Corrected'); ax2.axis('off')
            fig.tight_layout(); plt.show(block=False)
            self._status('Distortion correction done.')

    # --- Chessboard ---
    def _on_chess_cal(self, event=None):
        root = tk.Tk(); root.withdraw()
        path = filedialog.askdirectory(title='Select folder with chessboard images')
        root.destroy()
        if not path:
            return
        from .chessboard.calibration import calibrate_camera
        try:
            params = calibrate_camera(path)
            if params is None:
                return
            text = (f"Size: {params['image_size']}\n"
                    f"Focal: {params['focal_length']}\n"
                    f"Radial k: {params['radial_distortion']}\n"
                    f"Reproj err: {params['mean_reprojection_error']:.4f} px\n"
                    f"Images: {params['num_images_used']}")
        except Exception as e:
            text = f'Error: {e}'
        fig, ax = plt.subplots(figsize=(6, 4), num='Calibration Results')
        ax.text(0.1, 0.5, text, fontsize=11, va='center', family='monospace')
        ax.axis('off'); fig.tight_layout(); plt.show(block=False)
        self._status('Calibration complete.')

    # --- Color check ---
    def _compute_color(self):
        from .color_check import _color_check_from_rois
        delta_E, delta_C, sat = _color_check_from_rois(self.rois)
        self._status(f'dE mean={delta_E[24]:.2f} max={delta_E[25]:.2f}  '
                     f'dC mean={delta_C[24]:.2f}  sat={sat[24]*100:.1f}%')

    # --- Step chart ---
    def _compute_step(self):
        from .step_chart import _step_chart_from_rois
        n, ratio = _step_chart_from_rois(self.rois)
        self._status(f'{n} steps detected. Contrast ratio = {ratio:.2f}')

    # --- White balance ---
    def _compute_wb(self):
        from .white_balance import _wb_from_rois
        whiteB, piancha = _wb_from_rois(self.rois)
        if whiteB is not None:
            ch = {1: 'R', 2: 'G', 3: 'B'}
            wb_str = '  '.join([f'{w:.1f}' for w in whiteB])
            pc_str = ' '.join([ch.get(int(p), '?') for p in piancha])
            self._status(f'WB error: {wb_str}  |  Dominant: {pc_str}')

    # --- Dark field ---
    def _on_dark_field(self, event=None):
        if self.img is None:
            self._on_open()
            if self.img is None:
                return
        if self.img.ndim == 3:
            g = np.dot(self.img[..., :3], [0.2989, 0.5870, 0.1140])
        else:
            g = self.img
        self._status(f'Dark field: mean={np.mean(g):.1f}  std={np.std(g):.1f}  '
                     f'min={np.min(g):.0f}  max={np.max(g):.0f}')

    def run(self):
        plt.show()


def main():
    app = ImagingQualityApp()
    app.run()


if __name__ == '__main__':
    main()
