# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 13:26:07 2026

@author: Sean
"""
from pathlib import Path
import re
import numpy as np
import sys
from bokeh.plotting import figure, show
from bokeh.io import curdoc, save, output_file
from bokeh.palettes import d3
from bokeh.models import HoverTool, ColumnDataSource, DataTable, TableColumn
from bokeh.models import CrosshairTool, CustomJS, Div, Toggle, Button
from bokeh.layouts import column, row
curdoc().theme = "dark_minimal"
palette = d3["Category10"]


def phase_color(phase_number):
    return palette[10][phase_number % 10]


def twotheta_to_d(two_theta, wave):
    with np.errstate(divide="ignore"):
        return wave / (2 * np.sin(np.deg2rad(np.asarray(two_theta, dtype=float) / 2)))


def find_sub_files(prf_path):
    prf_path = Path(prf_path)
    stem = prf_path.stem
    patterns = [re.compile(re.escape(stem) + r"(\d+)\.sub", re.IGNORECASE)]
    bank = re.fullmatch(r"(.*)_(\d+)", stem)
    if bank:
        patterns.append(re.compile(re.escape(bank.group(1)) + r"(\d+)_" + bank.group(2) + r"\.sub", re.IGNORECASE))
    found = {}
    for sub in prf_path.resolve().parent.glob("*.sub"):
        for pattern in patterns:
            match = pattern.fullmatch(sub.name)
            if match:
                found[sub] = int(match.group(1))
                break
    return sorted(found, key=lambda s: (found[s], s.name))


def tof_to_d(tof, zero, dtt1, dtt2):
    t = np.asarray(tof, dtype=float) - zero
    with np.errstate(invalid="ignore", divide="ignore"):
        return 2 * t / (dtt1 + np.sqrt(dtt1**2 + 4 * dtt2 * t))


class PRF:
    def __init__(self, prf_path, partials_paths=None):
        self.prf_path = Path(prf_path)
        self.filename = self.prf_path.stem
        self.phase_dict = {}
        if partials_paths is None:
            self.partials_paths = self.detect_sub()
        else:
            self.partials_paths = [Path(p) for p in partials_paths]
            self.sub_case = len(self.partials_paths) > 0
        for path in self.partials_paths:
            self.read_sub(path)
        self.read(self.prf_path)

    def redo(self):
        self.phase_dict = {}
        for path in self.partials_paths:
            self.read_sub(path)
        self.read(self.prf_path)

    def detect_sub(self):
        # look in same folder as .prf file
        folder = self.prf_path.resolve().parent
        nl = "\n"
        all_subs = sorted(folder.glob("*.sub"))
        if all_subs == []:
            print("No .sub files found, recommend setting Ipr = 2 in .pcr for partials calculation")
            self.sub_case = False
            return []
        stem = self.prf_path.stem
        sub_files = find_sub_files(self.prf_path)
        if sub_files:
            print(f"Detected .sub files matching .prf name:\n{nl.join(i.stem for i in sub_files)}")
            use = input("Use all these files to plot partials? [Y]/n\nOr 'p' to pick over all .sub files in folder\nPress Enter for [Y]:\n").strip()
        else:
            print(f"No .sub files matching '{stem}' found, but other .sub files exist in {folder}")
            use = input("Enter 'p' to pick from them, anything else to skip partials:\n").strip()
        if sub_files and use in ("y", "Y", ""):
            print("Reading .sub files")
            self.sub_case = True
            return sub_files
        if use == "p":
            for index, file in enumerate(all_subs):
                print(f"{index} : {file}")
            indexs = input("Specify index of sub files to use: (eg: '0 1 2' for files 0, 1, 2)\nWARNING: assumes order is same as hkl data in .prf & phase order in .pcr\n").split()
            files = []
            for i in indexs:
                try:
                    files.append(all_subs[int(i)])
                except (ValueError, IndexError):
                    print(f"Ignoring invalid index '{i}'")
            self.sub_case = len(files) > 0
            return files
        print("Not reading .sub files")
        self.sub_case = False
        return []

    def read_sub(self, sub_path):
        with open(sub_path) as sub:
            sub_buffer = [line for line in sub if line.strip()]
        # header e.g. "! Phase No.:   1 Nuclear and Magnetic Structure"
        header_tokens = sub_buffer[0].split(":", 1)[1].split()
        phase_number = str(int(header_tokens[0]))
        phase_name = " ".join(header_tokens[1:]) or f"Phase {phase_number}"
        sub_data = np.array([[float(i) for i in line.split()[:2]] for line in sub_buffer[1:]])
        self.phase_dict[phase_number] = {
            "name" : [phase_name],
            "x" : sub_data[:,0],
            "disp_x" : sub_data[:,0],
            "d" : [],
            "y" : sub_data[:,1],
            "disp_y" : sub_data[:,1]
            }

    def x_to_d(self, x):
        if self.data_type == "tof":
            return tof_to_d(x, self.prf_header["zero"], self.prf_header["dtt1"], self.prf_header["dtt2"])
        return twotheta_to_d(x, self.prf_header["wave"][0])

    def read(self, prf_path):
        # file I/O
        header_index = None
        with open(prf_path) as prf:
            buffer = []
            for index, line in enumerate(prf):
                line = line.strip()
                buffer.append(line)
                # check for data type (first match only)
                if header_index is None and line.startswith("2Theta"):
                    header_index = index
                    self.data_type = "2theta"
                if header_index is None and line.startswith("T.O.F."):
                    header_index = index
                    self.data_type = "tof"
        if header_index is None:
            raise ValueError(f"{prf_path}: no '2Theta' or 'T.O.F.' column header found; is this a FullProf .prf file?")

        # read header
        header = [i.split() for i in buffer[1:header_index]]
        header_dict = {}
        header_dict["title"] = buffer[0].split("CELL")[0].strip()
        header_dict["phases"] = int(header[0][0])
        n_points = int(header[0][1])
        header_dict["excluded_regions"] = [
            np.array(i, dtype = "float") for i in header[2:2+int(header[1][-1])]
            ]
        header_dict["wave"] = [float(i) for i in header[0][2:4]]
        header_dict["zero"] = float(header[0][4])
        header_dict["dtt1"] = float(header[0][5])
        header_dict["dtt2"] = float(header[0][6])
        self.prf_header = header_dict
        n_phases = header_dict["phases"]

        # split into data
        raw_data = np.array([i.split()[:5] for i in buffer[header_index+1:header_index+1+n_points]], dtype = "float")
        if raw_data.shape[0] != n_points:
            raise ValueError(f"{prf_path}: header says {n_points} points but only {raw_data.shape[0]} found")
        self.x = raw_data[:,0]
        self.y_obs = raw_data[:,1]
        self.y_calc = raw_data[:,2]
        self.raw_diff = self.y_obs - self.y_calc
        self.plot_diff = raw_data[:,3]
        self.background = raw_data[:,4]

        def log10_clip(a):
            return np.log10(np.clip(a, a_min = 1e-5, a_max = None))

        # offset the difference curve below the data
        diff_offset = np.max(self.y_obs) * 0.3 / n_phases
        log_diff_offset = 0.3 * log10_clip(np.max(self.y_obs)) / n_phases
        diff = self.raw_diff - diff_offset
        log_diff = log10_clip(self.y_obs) - log10_clip(self.y_calc) - log_diff_offset

        # prepare dict for columnsource
        self.d = self.x_to_d(self.x)
        self.main_source = {
            "x" : self.x,
            "disp_x" : self.x,
            "d" : self.d,

            "y_obs" : self.y_obs,
            "log_y_obs" : log10_clip(self.y_obs),
            "disp_y_obs" : self.y_obs,

            "y_calc" : self.y_calc,
            "log_y_calc" : log10_clip(self.y_calc),
            "disp_y_calc" : self.y_calc,

            "diff" : diff,
            "log_diff" : log_diff,
            "disp_diff" : diff,

            "background" : self.background,
            "log_background" : log10_clip(self.background),
            "disp_background" : self.background
        }

        # excluded regions
        self.excluded_source = {}
        for index, pair in enumerate(header_dict["excluded_regions"]):
            d1, d2 = self.x_to_d(pair[:2])
            self.excluded_source[str(index)] = {
                "x1": [float(pair[0])],
                "x2": [float(pair[1])],
                "disp_x1": [float(pair[0])],
                "disp_x2": [float(pair[1])],
                "d1": [float(d1)],
                "d2": [float(d2)],
            }

        # read hkl data (skip blank lines)
        hkl_lines = [i for i in buffer[header_index+1+n_points:] if "(" in i and ")" in i]
        self.raw_hkl = [
                   [
                    int(i.split()[-1]),
                    i.split("(")[0].split()[0],
                    i.split("(")[1].split(")")[0].strip()
                    ]
                   for i in hkl_lines]
        self.raw_hkl = [i for i in self.raw_hkl if float(i[1]) <= np.max(self.x)]

        # prepare phase dicts, nested dict with partials data and name stored in main then hkl data stored in nested dict
        tick_base = np.average(self.y_obs - self.y_calc - np.max(self.y_obs) * n_phases * 0.1)
        for phase_number in range(1, n_phases+1):
            key = str(phase_number)
            #parses raw hkl data into per phase data
            hkl_x = np.array([float(i[1]) for i in self.raw_hkl if i[0] == phase_number])
            hkl_y = np.full(len(hkl_x), tick_base / n_phases * phase_number * 0.3)
            hkl_label = np.array([i[2] for i in self.raw_hkl if i[0] == phase_number])
            # calculated d
            if self.data_type == "2theta":
                d_1 = twotheta_to_d(hkl_x, header_dict["wave"][0])
                d_2 = twotheta_to_d(hkl_x, header_dict["wave"][1])
            else:
                d_1 = self.x_to_d(hkl_x)
                d_2 = d_1
            # constructs outer dict; phases with no .sub file get an empty partial
            if key not in self.phase_dict:
                self.phase_dict[key] = {
                    "name": [f"Phase {phase_number}"],
                    "x": [],
                    "disp_x" : [],
                    "d" : [],
                    "y": [],
                    "log_y": [],
                    "disp_y": []
                }
            else:
                pd = self.phase_dict[key]
                pd["d"] = self.x_to_d(pd["x"])
                #taking the log of the raw partials is incorrect, need to take log of partial + bkg
                if len(pd["x"]) == len(self.x) and np.allclose(pd["x"], self.x):
                    bkg = self.background
                else:
                    bkg = np.interp(pd["x"], self.x, self.background)
                pd["log_y"] = log10_clip(pd["y"] + bkg)
            phase_name = self.phase_dict[key]["name"][0]
            #constructs inner dict
            hkl_dict = [dict(
                name = [phase_name] * len(hkl_x),
                x = hkl_x,
                disp_x = hkl_x,
                y = hkl_y,
                log_y = -(np.log10(np.abs(hkl_y)+1e-5)),
                disp_y = hkl_y,
                label = hkl_label,
                d_1 = d_1,
                d_2 = d_2,
            )]
            self.phase_dict[key]["hkl_data"] = hkl_dict

    def plot_bokeh(self, show_plot=True):

        #main plot
        plotting_source = ColumnDataSource(self.main_source)
        plot = figure(
            sizing_mode="stretch_both",
            x_axis_label= self.data_type,
            y_axis_label="Intensity",
            title = f"{self.prf_header['title']}, wave1 = {self.prf_header['wave'][0]} Å, wave2 = {self.prf_header['wave'][1]} Å",
            )
        obs_plot = plot.scatter(
            x = "disp_x",
            y = "disp_y_obs",
            legend_label = "obs",
            color = "white",
            source = plotting_source
            )
        calc_plot = plot.line(
            x = "disp_x",
            y = "disp_y_calc",
            legend_label = "calc",
            color = "red",
            line_width = 2.5,
            source = plotting_source
            )
        diff_plot = plot.line(
            x = "disp_x",
            y = "disp_diff",
            legend_label = "diff",
            color = "lightgrey",
            line_width = 2.5,
            source = plotting_source
            )
        bkg_plot = plot.line(
            x = "disp_x",
            y = "disp_background",
            legend_label = "bkg",
            color = "yellow",
            line_width = 2.5,
            source = plotting_source
            )

        #partials plot
        partials_sources = []
        for phase_number in range(1, self.prf_header["phases"] + 1):
            #if the phase dict item is sent to the ColumnDataSource it will not work, assumes all dict values are arrays of equal length
            partials_source = ColumnDataSource(
                dict(
                    disp_x = self.phase_dict[str(phase_number)]["disp_x"],
                    disp_y =  self.phase_dict[str(phase_number)]["disp_y"],
                    x = self.phase_dict[str(phase_number)]["x"],
                    y = self.phase_dict[str(phase_number)]["y"],
                    d = self.phase_dict[str(phase_number)]["d"],
                    log_y = self.phase_dict[str(phase_number)]["log_y"]
                )
            )
            partials_sources.append(partials_source)
            plot.line(
                x = "disp_x",
                y = "disp_y",
                source = partials_sources[-1],
                legend_label = self.phase_dict[str(phase_number)]["name"][0],
                color = phase_color(phase_number),
                line_width = 2,
            )

        #hkl plots
        hkl_plots = []
        hkl_sources = []
        for number in range(1, self.prf_header["phases"]+1):
            phase_dict = self.phase_dict[str(number)]
            hkl_source = ColumnDataSource(phase_dict["hkl_data"][0])
            hkl_sources.append(hkl_source)
            r = plot.scatter(
                x="disp_x",
                y="disp_y",
                source=hkl_sources[-1],
                legend_label=phase_dict["name"][0],
                marker="dash",
                angle=np.pi / 2,
                fill_color=phase_color(number),
                line_color=phase_color(number),
                size=20
            )
            hkl_plots.append(r)

        #handles hovering over hkls
        x_tooltip = ("2theta (°)", "$snap_x") if self.data_type == "2theta" else ("TOF (us)", "$snap_x")
        hkl_hover = HoverTool(
            tooltips = [
                ("Phase", "@name"),
                x_tooltip,
                ("d (Å)", "@d_1"),
                ("hkl", "@label")
            ],
            renderers = hkl_plots
            )

        #adds tools to toolbar
        plot.add_tools(hkl_hover)
        cross_tool = CrosshairTool(line_color = "lightgrey")
        plot.add_tools(cross_tool)

        #plots excluded regions
        excluded_plots = []
        excluded_sources = []
        for key, value in self.excluded_source.items():
            excluded_source = ColumnDataSource(value)
            excluded_sources.append(excluded_source)
            e = plot.vstrip(
                x0 = "disp_x1",
                x1 = "disp_x2",
                hatch_pattern = "x",
                source = excluded_sources[-1],
                line_color = "black",
                alpha = 0.65,
                fill_color = "grey"
                )
            excluded_plots.append(e)
        #handle switching from y to log(y)
        toggle_log = Toggle(label="Switch to log10", button_type="default", active=False)
        update_log = CustomJS(args=dict(
            y_axis = plot.yaxis[0],
            main_source=plotting_source,
            partials_sources = partials_sources,
            hkl_sources = hkl_sources,
            toggle = toggle_log
        ), code="""
            if (toggle.active) {
                main_source.data['disp_y_obs'] = main_source.data['log_y_obs'];
                main_source.data['disp_y_calc'] = main_source.data['log_y_calc'];
                main_source.data['disp_diff'] = main_source.data['log_diff'];
                main_source.data['disp_background'] = main_source.data['log_background'];
                for (let i = 0; i < partials_sources.length; i++) {
                partials_sources[i].data['disp_y'] = partials_sources[i].data['log_y'];
                }
                for (let i = 0; i < hkl_sources.length; i++) {
                hkl_sources[i].data['disp_y'] = hkl_sources[i].data['log_y'];
                }
                y_axis.axis_label = "log10";
                toggle.label = "Switch to Intensity";
                toggle.button_type = "primary"

            } else {
                main_source.data['disp_y_obs'] = main_source.data['y_obs'];
                main_source.data['disp_y_calc'] = main_source.data['y_calc'];
                main_source.data["disp_diff"] = main_source.data["diff"];
                main_source.data['disp_background'] = main_source.data['background'];
                for (let i = 0; i < partials_sources.length; i++) {
                partials_sources[i].data['disp_y'] = partials_sources[i].data['y'];
                }

                for (let i = 0; i < hkl_sources.length; i++) {
                hkl_sources[i].data['disp_y'] = hkl_sources[i].data['y'];
                }
                y_axis.axis_label = "Intensity"
                toggle.label = "Switch to log10";
                toggle.button_type = "success"
            }
            main_source.change.emit();
            for (let i = 0; i < partials_sources.length; i++) {
            partials_sources[i].change.emit();
            }
            for (let i = 0; i < hkl_sources.length; i++) {
                    hkl_sources[i].change.emit();
                    }

        """)
        toggle_log.js_on_change('active', update_log)
        #handle x to d
        toggle_x = Toggle(label="Switch to d space", button_type="default", active=False)
        update_x = CustomJS(args=dict(main_source=plotting_source,
                                        partials_sources=partials_sources,
                                        hkl_sources=hkl_sources,
                                        excluded_sources = excluded_sources,
                                        toggle=toggle_x,
                                        label = self.data_type,
                                        variable = self.data_type,
                                        x_axis = plot.xaxis[0],
                                        ), code="""
                    if (toggle.active) {
                        main_source.data['disp_x'] = main_source.data['d'];
                        for (let i = 0; i < partials_sources.length; i++) {
                        partials_sources[i].data['disp_x'] = partials_sources[i].data['d'];
                        }

                        for (let i = 0; i < hkl_sources.length; i++) {
                        hkl_sources[i].data['disp_x'] = hkl_sources[i].data['d_1'];
                        }
                        for (let i = 0; i < excluded_sources.length; i++) {
                        excluded_sources[i].data['disp_x1'] = excluded_sources[i].data['d1'];
                        excluded_sources[i].data['disp_x2'] = excluded_sources[i].data['d2'];
                        }
                        x_axis.axis_label = "d space (A)";
                        toggle.label = "Switch to "+variable;
                        toggle.button_type = "primary"
                    } else {
                        main_source.data['disp_x'] = main_source.data['x'];
                        for (let i = 0; i < partials_sources.length; i++) {
                        partials_sources[i].data['disp_x'] = partials_sources[i].data['x'];
                        }

                        for (let i = 0; i < hkl_sources.length; i++) {
                        hkl_sources[i].data['disp_x'] = hkl_sources[i].data['x'];
                        }
                        for (let i = 0; i < excluded_sources.length; i++) {
                        excluded_sources[i].data['disp_x1'] = excluded_sources[i].data['x1'];
                        excluded_sources[i].data['disp_x2'] = excluded_sources[i].data['x2'];
                        }
                        x_axis.axis_label = label;
                        toggle.label = "Switch to d space";

                        toggle.button_type = "success"
                    }
                    main_source.change.emit();
                    for (let i = 0; i < partials_sources.length; i++) {
                    partials_sources[i].change.emit();
                    }
                    for (let i = 0; i < excluded_sources.length; i++) {
                    excluded_sources[i].change.emit();
                    }

                    for (let i = 0; i < hkl_sources.length; i++) {
                    hkl_sources[i].change.emit();
                    }
                """)
        toggle_x.js_on_change('active', update_x)
        #coordinates bar at the bottom
        coords = Div(
            text="x: -, d: -, y: -",
            sizing_mode="stretch_width",
        )
        # JS x -> d conversion
        if self.data_type == "2theta":
            d_args = dict(wave = float(self.prf_header["wave"][0]))
            d_js = """
            function x_to_d(x) {
                return wave / (2 * Math.sin(x * Math.PI / 360.0));
            }
            """
            x_text = "2theta"
            x_unit = "°"
        else:
            d_args = dict(
                c0=self.prf_header["zero"],
                c1=self.prf_header["dtt1"],
                c2=self.prf_header["dtt2"]
            )
            d_js = """
            // invert TOF = c0 + c1*d + c2*d^2 (stable form, physical root for either sign of c2)
            function x_to_d(x) {
                const t = x - c0;
                const disc = c1 * c1 + 4 * c2 * t;
                if (disc < 0) { return NaN; }
                const denom = c1 + Math.sqrt(disc);
                return denom === 0 ? NaN : 2 * t / denom;
            }
            """
            x_text = "Time-of-Flight (TOF)"
            x_unit = "μs"

        plot.js_on_event("mousemove", CustomJS(args=dict(div=coords, **d_args), code=d_js + f"""
            const x = cb_obj.x;
            const y = cb_obj.y;
            const d = x_to_d(x);
            const d_text = isNaN(d) ? "N/A" : d.toFixed(5);
            div.text = `<b>{x_text}: ${{x.toFixed(2)}} {x_unit}, d: ${{d_text}} Å, y: ${{y.toFixed(2)}}</b> - click to store values`;
            """))

        #draw a point where you click on the plot
        clicked_source = ColumnDataSource(data=dict(x=[], y=[], d=[], label=[]))
        clear_button = Button(label="Clear points", button_type = "danger", sizing_mode = "stretch_width")
        clear_button.js_on_event("button_click", CustomJS(args = dict(
            button = clear_button,
            data_source = clicked_source
        ),
        code = """
        data_source.data["x"] = [];
        data_source.data["y"] = [];
        data_source.data["d"] = [];
        data_source.data["label"] = [];
        data_source.change.emit()
        """
        ))

        plot.scatter(
            x="x",
            y="y",
            source=clicked_source,
            marker="x",
            size = 20,
            line_width = 2,
            color = "deepskyblue"
            )
        plot.text(
            x="x",
            y="y",
            text="label",
            source=clicked_source,
            text_color="deepskyblue",
            text_font_size="10pt",
            y_offset=-15
            )
        #stick that point in a table
        plot.js_on_event(
            "tap",
            CustomJS(
                args=dict(source=clicked_source, **d_args),
                code=d_js + """
                const data = source.data;
                const x = cb_obj.x;
                const y = cb_obj.y;
                const d = x_to_d(x);
                const next_number = String(data['x'].length + 1);  // "1", "2", "3"...
                data['x'].push(Number(x.toFixed(5)));
                data['y'].push(Number(y.toFixed(5)));
                data['d'].push(isNaN(d) ? NaN : Number(d.toFixed(5)));
                if (!data['label']) { data['label'] = []; }
                data['label'].push(next_number);
                source.change.emit();
                """
            )
        )
        #button to download table:
        download_button = Button(label="Download table", button_type="success")
        save_table = """
        const data = source.data;
        const columns = Object.keys(data);

        let out_data = columns.join(",") + "\\n";

        const numRows = data[columns[0]].length;

        for (let i = 0; i < numRows; i++) {
            let row = columns.map(col => {
                let value = data[col][i];
                return value;
            });
            out_data += row.join(",") + "\\n";
        }

        const blob = new Blob([out_data], { type: 'text/csv;charset=utf-8;' });

        const link = document.createElement("a");
        const url = URL.createObjectURL(blob);
        link.setAttribute("href", url);
        link.setAttribute("download", "bokeh_table_data.csv");
        link.style.visibility = 'hidden';
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        """

        download_button.js_on_click(CustomJS(args=dict(source=clicked_source), code=save_table))

        #do the GUI elements
        columns = [
            TableColumn(field="x", title="x"),
            TableColumn(field="d", title="d-spacing (Å)"),
            TableColumn(field="y", title="y"),
        ]

        table = DataTable(
            source=clicked_source,
            columns=columns,
            sizing_mode="stretch_both",
        )

        plot_area = column(
            plot,
            coords,
            sizing_mode="stretch_both"
        )

        control_area = column(
            row(toggle_x, toggle_log, sizing_mode = "stretch_width"),
            table,
            row(download_button, clear_button),
            width = 300,
            sizing_mode = "stretch_height"
        )

        layout = row(
            control_area,
            plot_area,
            sizing_mode="stretch_both"
            )

        if show_plot:
            show(layout)
        return layout


def resolve_prf_path(arg):
    path = Path(arg)
    if path.suffix.lower() != ".prf":
        if path.suffix != "":
            raise ValueError(f"Not a valid .prf file: {arg}")
        path = path.with_suffix(".prf")
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")
    return path


def choose_prf(folder):
    prfs = sorted(Path(folder).glob("*.prf"))
    if prfs == []:
        print(f"No .prf files found in {folder}")
        return None
    if len(prfs) == 1:
        print(f"Found .prf file: {prfs[0]}\nUsing this file")
        return prfs[0]
    print(f"Detected {len(prfs)} .prf files:")
    for index, i in enumerate(prfs):
        print(f"{index} : {i}")
    while True:
        num = input(f"Specify index of .prf files to show: (0 to {len(prfs)-1})\n").strip()
        if num.isdigit() and int(num) < len(prfs):
            return prfs[int(num)]
        print("Invalid index, try again")


def interactive_session(prf):
    out = prf.plot_bokeh()
    while True:
        save_html = input("\nPress Enter to exit, r to reload .prf, or enter a filename to save .html\n").strip()
        if save_html == "":
            print("No .html saved")
            return None
        elif save_html == "r":
            prf.redo()
            out = prf.plot_bokeh()
        else:
            filename = save_html if save_html.lower().endswith(".html") else f"{save_html}.html"
            output_file(filename=filename, title=prf.prf_header["title"])
            save(out)
            print(f"Saved {filename}")
            return filename


def main(argv=None):
    argv = sys.argv if argv is None else argv
    print(f"Running {argv[0] if argv else 'PRF.py'}\n")
    print("THIS SCRIPT WILL GENERATE AN INTERACTABLE WEBPAGE FOR ANALYSIS OF A FULLPROF .PRF FILE\n"
          "TESTED WITH X-RAY AND TOF DATA\n"
          "THE WEBPAGE IS GENERATED AND STORED LOCALLY AND MAY BE SAVED AS A .HTML FILE\n"
          "WRITTEN BY S. INJAC\n\n\n")
    args = argv[1:]
    if len(args) == 0:
        print("To use this script, pass the path of a FullProf .prf file as an argument\n")
        path = choose_prf(Path.cwd())
        if path is None:
            return 1
    elif len(args) == 1:
        try:
            path = resolve_prf_path(args[0])
        except (ValueError, FileNotFoundError) as e:
            print(e)
            return 1
    else:
        print("Usage: python PRF.py [file.prf]")
        return 1
    interactive_session(PRF(path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
