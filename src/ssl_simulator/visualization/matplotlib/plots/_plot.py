import logging
import os

import matplotlib.pyplot as plt
import numpy as np
from IPython.display import HTML
from matplotlib.animation import FuncAnimation, PillowWriter
from tqdm import tqdm

from ssl_simulator.logging import requires_log_level
from ssl_simulator.utils.path_ops import create_dir

_logger = logging.getLogger(__name__)


class PlotBase:
    """
    Generic base class for all plot types.

    Subclasses should:
        - define self.axes_config: dict specifying axes
        - implement init_artists(self)
        - implement update_artists(self, frame_data)
        - override compute_frame(self, frame_idx) if needed for large logs
    """

    def __init__(self, figsize=(8, 6), dpi=100):
        self.fig = plt.figure(figsize=figsize, dpi=dpi)
        self.axes = {}
        self.artists = {}
        self.artists_list = []
        self._initialized = False
        self.figsize = figsize
        self.dpi = dpi

        # To be defined by subclass
        self.axes_config = {}  # e.g., {"main": {"positions":[0,0,1,1], "projection":"3d"}}
        self.n_frames = 0  # total number of frames

    # ---------------------------------------------------------------
    # SETUP AND INITIALIZATION
    # ---------------------------------------------------------------
    def setup_axes(self):
        """Create axes based on self.axes_config."""
        for key, cfg in self.axes_config.items():
            cfg_copy = cfg.copy()  # avoid modifying the original config

            # Extract rect/position
            rect = None
            if "position" in cfg_copy:
                rect = cfg_copy.pop("position")
            if "rect" in cfg_copy:
                if rect is not None:
                    raise ValueError(f"Axis '{key}': Cannot provide both 'rect' and 'position'.")
                rect = cfg_copy.pop("rect")
            if rect is None:
                raise ValueError(f"Axis '{key}': Must provide either 'position' or 'rect'.")

            # Create axis with remaining kwargs (e.g., projection)
            self.axes[key] = self.fig.add_axes(rect, **cfg_copy)
            self.artists[key] = {}

            _logger.debug(f"Created axis '{key}' with rect={rect} and kwargs={cfg_copy}")

    def init_artists(self):
        """Initialize all plot elements. Must be implemented by subclass."""
        raise NotImplementedError

    def setup(self):
        """Full setup: create axes and initialize artists."""
        if not self._initialized:
            self.setup_axes()
            self.init_artists()
            self._initialized = True

            def flatten_artists(group):
                """Recursively yield all artist objects in group (dict, list, ndarray, single object)."""
                if isinstance(group, dict):
                    for v in group.values():
                        yield from flatten_artists(v)
                elif isinstance(group, list):
                    for item in group:
                        yield from flatten_artists(item)
                elif isinstance(group, np.ndarray):
                    for item in group.flat:
                        yield item
                else:
                    yield group

            self.artists_list = list(flatten_artists(self.artists))
            # Use setattr to avoid static type issues with FigureCanvasBase
            self.fig.canvas.draw_count = 0  # reset draw count

    # ---------------------------------------------------------------
    # UPDATE + ANIMATION CORE
    # ---------------------------------------------------------------
    def update_artists(self, frame_data):
        """Update plot elements for a new frame. Must be implemented by subclass."""
        raise NotImplementedError

    def update(self, frame_data):
        """Wrapper called by FuncAnimation or manual updates."""
        self.update_artists(frame_data)
        self.anim_progress.update(1)
        # return list of all artists for blitting
        return []

    def compute_frame(self, frame_idx):
        """Compute any data needed for the given frame index."""
        return frame_idx

    def data_generator(self, precompute=False):
        """
        Yields frames for animation, either on-the-fly or precomputed.

        Args:
            precompute (bool): If True, precompute all frames and store in memory.
                            If False, generate frames on-the-fly.

        Yields
        ------
            frame data (could be index or actual frame content depending on `compute_frame`)
        """
        if precompute:
            # Precompute all frames and store in a list
            frames = [
                self.compute_frame(i)
                for i in tqdm(range(self.n_frames), desc="Precomputing frames")
            ]
            # Use a separate progress bar for playback
            yield from tqdm(frames, desc="Animating frames")
        else:
            # Generate frames one by one
            for i in tqdm(range(self.n_frames), desc="Generating frames on-the-fly"):
                yield self.compute_frame(i)

    def animate(self, interval=100, repeat=False, blit=False):
        """Generic animation method."""
        self.setup()
        self.anim_progress = tqdm(total=self.n_frames, desc="Animating frames")
        frames = range(self.n_frames)
        ani = FuncAnimation(
            self.fig,
            self.update,
            frames=frames,
            interval=interval,  # 1/fps*1000/4
            blit=blit,
            repeat=repeat,
        )
        return ani

    # ---------------------------------------------------------------
    # VISUALIZATION AND EXPORT
    # ---------------------------------------------------------------
    def show(self):
        """Interactive view in a GUI backend."""
        self.setup()
        plt.show()

    def to_html(self, interval=100, repeat=False, blit=False, method="jshtml"):
        """
        Display animation inline in a Jupyter notebook.
        method: "jshtml" (default) or "html5" for video.
        """
        ani = self.animate(interval=interval, repeat=repeat, blit=blit)
        plt.close(self.fig)  # avoid double display
        if method == "html5":
            return HTML(ani.to_html5_video())
        return HTML(ani.to_jshtml())

    def save_gif(self, output_dir, filename="animation.gif", fps=30, interval=100, repeat=False):
        """Render animation and save as GIF."""
        create_dir(output_dir)  # ensure directory exists
        filepath = os.path.join(output_dir, filename)

        ani = self.animate(interval=interval, repeat=repeat)
        writer = PillowWriter(fps=fps)
        _logger.info(f"Saving GIF to {filename} ...")
        ani.save(filepath, writer=writer)
        _logger.info("GIF saved successfully.")

    def save_mp4(self, output_dir, filename="animation.mp4", fps=30, interval=100, repeat=False):
        """Render animation and save as MP4 (requires ffmpeg)."""
        create_dir(output_dir)  # ensure directory exists
        filepath = os.path.join(output_dir, filename)

        ani = self.animate(interval=interval, repeat=repeat)
        _logger.info(f"Saving MP4 to {filename} ...")
        ani.save(filepath, writer="ffmpeg", fps=fps)
        _logger.info("MP4 saved successfully.")

    # ---------------------------------------------------------------
    # DEBUG UTILITIES
    # ---------------------------------------------------------------
    @requires_log_level(_logger, logging.DEBUG)
    def debug_artists(self):
        """
        Print all artists in each axis with their key and type.
        Useful to inspect plot elements during development.
        """
        _logger.debug("----- PlotBase Artists Debug -----")
        total_count = 0

        def count_and_print(artist, prefix=""):
            """Recursively print and count artists."""
            nonlocal total_count
            if isinstance(artist, dict):
                for k, v in artist.items():
                    count_and_print(v, prefix=f"{prefix}[{k}]")
            elif isinstance(artist, list):
                for i, a in enumerate(artist):
                    count_and_print(a, prefix=f"{prefix}[{i}]")
            elif isinstance(artist, np.ndarray):
                _logger.debug(
                    f"{prefix} np.ndarray shape={artist.shape}, dtype={type(artist.flat[0]).__name__}"
                )
                total_count += artist.size
            else:
                _logger.debug(f"{prefix} {type(artist).__name__}: {artist}")
                total_count += 1

        for ax_key, _ax in self.axes.items():
            _logger.debug(f"Axis '{ax_key}' ({type(_ax).__name__}):")
            group = self.artists.get(ax_key, None)
            if not group:
                _logger.debug(f"    [!] No artists in axis '{ax_key}'")
                continue
            count_and_print(group, prefix="    ")

        _logger.debug(f"Total artists: {total_count}")
        _logger.debug("---------------------------------")
