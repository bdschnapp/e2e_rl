"""Tkinter launcher for simulation, training, and evaluation subprocesses.

The GUI does not embed pygame. It writes a structured JSON config, then starts
the selected entry point as a child process with the current environment so X11
forwarding keeps working over SSH.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import threading
import tkinter as tk
from dataclasses import fields
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText

from sim_config import (
    EVAL_FIELDS,
    RUN_MODEL_FIELDS,
    TRAIN_FIELDS,
    EvalConfig,
    FieldSpec,
    RunModelConfig,
    TrainConfig,
    reward_choices_for_scenario,
    validate_eval_config,
    validate_run_model_config,
    validate_train_config,
)


REPO_ROOT = Path(__file__).resolve().parent


class ConfigTab(ttk.Frame):
    def __init__(
        self,
        master,
        *,
        title: str,
        config_cls,
        field_specs: tuple[FieldSpec, ...],
        script_name: str,
        validator,
        output_callback,
        status_callback,
    ):
        super().__init__(master)
        self.title = title
        self.config_cls = config_cls
        self.field_specs = field_specs
        self.script_name = script_name
        self.validator = validator
        self.output_callback = output_callback
        self.status_callback = status_callback
        self.variables: dict[str, tk.Variable] = {}
        self.widgets: dict[str, tk.Widget] = {}
        self.current_config_path: Path | None = None
        self.current_process: subprocess.Popen[str] | None = None

        self.columnconfigure(0, weight=3)
        self.columnconfigure(1, weight=2)
        self.rowconfigure(1, weight=1)

        self._build_form()
        self._build_effective_config()
        self._load_defaults()
        self.refresh_effective_config()

    def _build_form(self):
        form = ttk.LabelFrame(self, text=self.title)
        form.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)
        form.columnconfigure(1, weight=1)

        for row_idx, spec in enumerate(self.field_specs):
            ttk.Label(form, text=spec.label).grid(row=row_idx, column=0, sticky="w", padx=6, pady=4)
            widget = self._create_widget(form, spec, row_idx)
            self.widgets[spec.name] = widget

        button_row = len(self.field_specs)
        button_frame = ttk.Frame(form)
        button_frame.grid(row=button_row, column=0, columnspan=3, sticky="ew", padx=6, pady=(10, 6))
        for idx in range(5):
            button_frame.columnconfigure(idx, weight=1)

        ttk.Button(button_frame, text="Run", command=self.run_subprocess).grid(row=0, column=0, sticky="ew", padx=2)
        ttk.Button(button_frame, text="Save Config", command=self.save_config_dialog).grid(row=0, column=1, sticky="ew", padx=2)
        ttk.Button(button_frame, text="Load Config", command=self.load_config_dialog).grid(row=0, column=2, sticky="ew", padx=2)
        ttk.Button(button_frame, text="Show Effective Config", command=self.refresh_effective_config).grid(row=0, column=3, sticky="ew", padx=2)
        ttk.Button(button_frame, text="Stop", command=self.stop_subprocess).grid(row=0, column=4, sticky="ew", padx=2)

    def _build_effective_config(self):
        box = ttk.LabelFrame(self, text="Effective Config")
        box.grid(row=0, column=1, sticky="nsew", padx=(0, 8), pady=8)
        box.rowconfigure(0, weight=1)
        box.columnconfigure(0, weight=1)
        self.config_text = ScrolledText(box, width=50, height=26, wrap="word")
        self.config_text.grid(row=0, column=0, sticky="nsew", padx=6, pady=6)

    def _create_widget(self, parent, spec: FieldSpec, row_idx: int):
        default_obj = self.config_cls()
        default_value = getattr(default_obj, spec.name)

        if spec.field_type == "bool":
            var = tk.BooleanVar(value=bool(default_value))
            widget = ttk.Checkbutton(parent, variable=var, command=self._on_form_change)
            widget.grid(row=row_idx, column=1, sticky="w", padx=6, pady=4)
            self.variables[spec.name] = var
            return widget

        if spec.field_type in {"choice", "choice_optional"}:
            values = list(spec.choices or ())
            if spec.field_type == "choice_optional":
                values = [""] + values
            var = tk.StringVar(value="" if default_value is None else str(default_value))
            widget = ttk.Combobox(parent, textvariable=var, values=values, state="readonly")
            widget.grid(row=row_idx, column=1, sticky="ew", padx=6, pady=4)
            widget.bind("<<ComboboxSelected>>", lambda _event: self._on_form_change())
            self.variables[spec.name] = var
            return widget

        if spec.field_type == "int":
            var = tk.StringVar(value=str(default_value))
            widget = ttk.Spinbox(
                parent,
                from_=spec.min_value if spec.min_value is not None else -1_000_000,
                to=spec.max_value if spec.max_value is not None else 1_000_000_000,
                textvariable=var,
            )
            widget.grid(row=row_idx, column=1, sticky="ew", padx=6, pady=4)
            widget.bind("<KeyRelease>", lambda _event: self._on_form_change())
            self.variables[spec.name] = var
            return widget

        var = tk.StringVar(value="" if default_value is None else str(default_value))
        entry = ttk.Entry(parent, textvariable=var)
        entry.grid(row=row_idx, column=1, sticky="ew", padx=6, pady=4)
        entry.bind("<KeyRelease>", lambda _event: self._on_form_change())
        self.variables[spec.name] = var
        if spec.field_type == "path":
            ttk.Button(parent, text="Browse", command=lambda name=spec.name: self._browse_path(name)).grid(
                row=row_idx, column=2, sticky="ew", padx=(0, 6), pady=4
            )
        return entry

    def _browse_path(self, field_name: str):
        current = self.variables[field_name].get()
        initialdir = str(REPO_ROOT if not current else Path(current).expanduser().parent)
        if field_name == "output_csv":
            selected = filedialog.asksaveasfilename(
                parent=self,
                initialdir=initialdir,
                title=f"Select {field_name}",
                defaultextension=".csv",
                filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")],
            )
        else:
            selected = filedialog.askopenfilename(
                parent=self,
                initialdir=initialdir,
                title=f"Select {field_name}",
            )
        if selected:
            self.variables[field_name].set(selected)
            self._on_form_change()

    def _load_defaults(self):
        self._update_dynamic_fields()

    def _on_form_change(self):
        self._update_dynamic_fields()
        self.refresh_effective_config()

    def _update_dynamic_fields(self):
        scenario = self.variables["scenario"].get() if "scenario" in self.variables else ""
        obs = self.variables["obs"].get() if "obs" in self.variables else ""

        reward_widget = self.widgets.get("reward")
        if isinstance(reward_widget, ttk.Combobox) and scenario:
            valid_rewards = list(reward_choices_for_scenario(scenario))
            reward_widget.configure(values=valid_rewards)
            if self.variables["reward"].get() not in valid_rewards:
                self.variables["reward"].set(valid_rewards[0])

        encoder_enabled = obs == "bev"
        lidar_enabled = obs == "lidar"
        for name in ("encoder", "encoder_path"):
            widget = self.widgets.get(name)
            if not widget:
                continue
            if isinstance(widget, ttk.Combobox):
                widget.configure(state="readonly" if encoder_enabled else "disabled")
            else:
                widget.configure(state="normal" if encoder_enabled else "disabled")
            if not encoder_enabled and name == "encoder":
                self.variables["encoder"].set("scratch")
            if not encoder_enabled and name == "encoder_path":
                self.variables["encoder_path"].set("")

        lidar_widget = self.widgets.get("lidar_beams")
        if lidar_widget:
            lidar_widget.configure(state="normal" if lidar_enabled else "disabled")

    def _collect_values(self):
        values = {}
        for dc_field in fields(self.config_cls):
            raw = self.variables[dc_field.name].get()
            if isinstance(self.variables[dc_field.name], tk.BooleanVar):
                values[dc_field.name] = bool(raw)
                continue
            if dc_field.type == int:
                values[dc_field.name] = int(raw)
                continue
            if dc_field.type == Path | None:
                values[dc_field.name] = Path(raw) if raw else None
                continue
            values[dc_field.name] = raw if raw != "" else None
        return values

    def build_config(self):
        config = self.config_cls.from_dict(self._collect_values())
        errors = self.validator(config)
        if errors:
            raise ValueError("\n".join(errors))
        return config

    def refresh_effective_config(self):
        try:
            config = self.config_cls.from_dict(self._collect_values())
            data = config.to_dict()
        except Exception as exc:
            data = {"error": str(exc)}
        self.config_text.delete("1.0", tk.END)
        self.config_text.insert(tk.END, json.dumps(data, indent=2, sort_keys=True, default=str))

    def save_config_dialog(self):
        try:
            config = self.build_config()
        except Exception as exc:
            messagebox.showerror("Invalid Config", str(exc), parent=self)
            return

        path = filedialog.asksaveasfilename(
            parent=self,
            title=f"Save {self.title} Config",
            defaultextension=".json",
            filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")],
            initialdir=str(REPO_ROOT),
        )
        if not path:
            return
        config.save_json(Path(path))
        self.current_config_path = Path(path)
        self.status_callback(f"Saved config: {path}")
        self.refresh_effective_config()

    def load_config_dialog(self):
        path = filedialog.askopenfilename(
            parent=self,
            title=f"Load {self.title} Config",
            filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")],
            initialdir=str(REPO_ROOT),
        )
        if not path:
            return

        config = self.config_cls.load_json(Path(path))
        for dc_field in fields(self.config_cls):
            value = getattr(config, dc_field.name)
            if isinstance(self.variables[dc_field.name], tk.BooleanVar):
                self.variables[dc_field.name].set(bool(value))
            else:
                self.variables[dc_field.name].set("" if value is None else str(value))
        self.current_config_path = Path(path)
        self._on_form_change()
        self.status_callback(f"Loaded config: {path}")

    def run_subprocess(self):
        if self.current_process and self.current_process.poll() is None:
            messagebox.showerror("Process Running", "A subprocess is already running for this tab.", parent=self)
            return

        try:
            config = self.build_config()
        except Exception as exc:
            messagebox.showerror("Invalid Config", str(exc), parent=self)
            return

        temp_file = tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".json",
            prefix=f"{self.script_name.replace('.py', '')}_",
            delete=False,
            dir="/tmp",
        )
        temp_path = Path(temp_file.name)
        temp_file.close()
        config.save_json(temp_path)

        command = [sys.executable, str(REPO_ROOT / self.script_name), "--config", str(temp_path)]
        env = os.environ.copy()
        self.output_callback(f"\n$ {' '.join(command)}\n")
        self.output_callback(f"[launcher] DISPLAY={env.get('DISPLAY', '<unset>')}\n")
        self.output_callback(f"[launcher] config={temp_path}\n")
        self.refresh_effective_config()

        self.current_process = subprocess.Popen(
            command,
            cwd=str(REPO_ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        self.status_callback(f"Running {self.script_name} ...")
        threading.Thread(target=self._stream_process_output, args=(self.current_process,), daemon=True).start()

    def _stream_process_output(self, process: subprocess.Popen[str]):
        assert process.stdout is not None
        for line in process.stdout:
            self.output_callback(line)
        return_code = process.wait()
        self.current_process = None
        self.status_callback(f"{self.script_name} exited with code {return_code}")
        self.output_callback(f"[launcher] process exited with code {return_code}\n")

    def stop_subprocess(self):
        if not self.current_process or self.current_process.poll() is not None:
            self.status_callback("No running subprocess to stop.")
            return
        self.current_process.terminate()
        self.output_callback("[launcher] sent SIGTERM to subprocess\n")
        self.status_callback(f"Stopping {self.script_name} ...")


class LauncherApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("E2E-RL Launcher")
        self.geometry("1400x900")
        self.tab_icons = self._build_tab_icons()

        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=3)
        self.rowconfigure(1, weight=2)

        notebook = ttk.Notebook(self)
        notebook.grid(row=0, column=0, sticky="nsew")

        self.status_var = tk.StringVar(value="Idle")

        self.run_tab = ConfigTab(
            notebook,
            title="Run Simulation",
            config_cls=RunModelConfig,
            field_specs=RUN_MODEL_FIELDS,
            script_name="run_model.py",
            validator=validate_run_model_config,
            output_callback=self.append_output,
            status_callback=self.set_status,
        )
        self.train_tab = ConfigTab(
            notebook,
            title="Train Model",
            config_cls=TrainConfig,
            field_specs=TRAIN_FIELDS,
            script_name="train.py",
            validator=validate_train_config,
            output_callback=self.append_output,
            status_callback=self.set_status,
        )
        self.eval_tab = ConfigTab(
            notebook,
            title="Evaluate Model",
            config_cls=EvalConfig,
            field_specs=EVAL_FIELDS,
            script_name="eval.py",
            validator=validate_eval_config,
            output_callback=self.append_output,
            status_callback=self.set_status,
        )

        notebook.add(
            self.run_tab,
            text=" Run Simulation",
            image=self.tab_icons["run_model"],
            compound="left",
        )
        notebook.add(
            self.train_tab,
            text=" Train",
            image=self.tab_icons["train"],
            compound="left",
        )
        notebook.add(
            self.eval_tab,
            text=" Eval",
            image=self.tab_icons["eval"],
            compound="left",
        )

        output_frame = ttk.LabelFrame(self, text="Subprocess Output")
        output_frame.grid(row=1, column=0, sticky="nsew", padx=8, pady=(0, 8))
        output_frame.columnconfigure(0, weight=1)
        output_frame.rowconfigure(0, weight=1)
        self.output_text = ScrolledText(output_frame, wrap="word", height=18)
        self.output_text.grid(row=0, column=0, sticky="nsew", padx=6, pady=6)

        status_bar = ttk.Label(self, textvariable=self.status_var, anchor="w")
        status_bar.grid(row=2, column=0, sticky="ew", padx=8, pady=(0, 8))

    def append_output(self, text: str):
        self.after(0, self._append_output_main_thread, text)

    def _append_output_main_thread(self, text: str):
        self.output_text.insert(tk.END, text)
        self.output_text.see(tk.END)

    def set_status(self, text: str):
        self.after(0, self.status_var.set, text)

    def _build_tab_icons(self) -> dict[str, tk.PhotoImage]:
        return {
            "run_model": self._make_letter_icon("#2e7d32", "R"),
            "train": self._make_letter_icon("#1565c0", "T"),
            "eval": self._make_letter_icon("#ef6c00", "E"),
        }

    def _make_letter_icon(self, color: str, letter: str) -> tk.PhotoImage:
        img = tk.PhotoImage(width=16, height=16)
        img.put(color, to=(0, 0, 16, 16))
        # Add a thin dark border so the icon stays visible on light themes.
        border = "#263238"
        for x in range(16):
            img.put(border, (x, 0))
            img.put(border, (x, 15))
        for y in range(16):
            img.put(border, (0, y))
            img.put(border, (15, y))

        # Simple block-letter glyphs drawn directly into the bitmap to avoid
        # any dependency on external image assets or font rendering.
        pixels = {
            "R": (
                (4, 3, 4, 12), (5, 3, 5, 12), (6, 3, 6, 12),
                (7, 3, 10, 3), (7, 7, 10, 7), (10, 4, 10, 6),
                (7, 8, 10, 12), (8, 8, 10, 10),
            ),
            "T": (
                (3, 3, 12, 3), (3, 4, 12, 4),
                (7, 3, 8, 12),
            ),
            "E": (
                (4, 3, 4, 12), (5, 3, 5, 12),
                (6, 3, 11, 3), (6, 7, 10, 7), (6, 11, 11, 11),
                (6, 12, 11, 12),
            ),
        }
        fg = "#ffffff"
        for x0, y0, x1, y1 in pixels[letter]:
            img.put(fg, to=(x0, y0, x1 + 1, y1 + 1))
        return img


def main():
    app = LauncherApp()
    app.mainloop()


if __name__ == "__main__":
    main()
