"""TablesGenerator — produce comparison tables matching baseline paper format.

Generates:
  Table 1  : F1 (%) across datasets
  Table 5  : Average token cost per query (milli-USD)
  Table 6  : Average retrieval call times per query
  Table 7  : Average total turns per query
  Table A  : Per-agent token cost (Table 3 equivalent)
  Table B  : Per-workflow turn cost (Table 4 equivalent)
  Table C  : CONDOR per-mechanism F1 breakdown (new)

All tables are saved as:
  - CSV  (machine-readable)
  - LaTeX (paper-ready, with bold/underline formatting)
  - Markdown (quick terminal inspection)
"""

import json
import os
import csv
from typing import Any, Dict, List, Optional

# ── Baseline numbers from paper (for reference / side-by-side) ───────────────
# Table 1: F1 (%)
BASELINE_F1 = {
    'LLM w/o RAG':       {'NQ': 39.96, 'PopQA': 30.99, 'AmbigQA': 49.90, 'HotpotQA': 42.38,
                           '2Wiki': 33.49, 'Musique': 20.74, 'Bamboogle': 36.15, 'Avg': 36.23},
    'Vanilla RAG':        {'NQ': 48.02, 'PopQA': 44.23, 'AmbigQA': 59.04, 'HotpotQA': 49.54,
                           '2Wiki': 37.62, 'Musique': 25.66, 'Bamboogle': 43.45, 'Avg': 43.94},
    'MAO-ARAG':           {'NQ': 54.50, 'PopQA': 54.16, 'AmbigQA': 57.80, 'HotpotQA': 53.80,
                           '2Wiki': 47.69, 'Musique': 37.33, 'Bamboogle': 65.09, 'Avg': 52.91},
}

# Table 5: Token cost (milli-USD)
BASELINE_TOKEN_COST = {
    'LLM w/o RAG':  {'NQ': 0.087, 'PopQA': 0.083, 'AmbigQA': 0.085, 'HotpotQA': 0.089,
                     '2Wiki': 0.088, 'Musique': 0.088, 'Bamboogle': 0.086, 'Avg': 0.086},
    'Vanilla RAG':  {'NQ': 0.258, 'PopQA': 0.265, 'AmbigQA': 0.256, 'HotpotQA': 0.265,
                     '2Wiki': 0.271, 'Musique': 0.263, 'Bamboogle': 0.258, 'Avg': 0.262},
    'MAO-ARAG':     {'NQ': 0.396, 'PopQA': 0.396, 'AmbigQA': 0.387, 'HotpotQA': 1.842,
                     '2Wiki': 1.633, 'Musique': 1.735, 'Bamboogle': 1.515, 'Avg': 1.129},
}

# Table 6: Retrieval calls
BASELINE_RETRIEVAL = {
    'LLM w/o RAG':  {'NQ': 0,   'PopQA': 0,   'AmbigQA': 0,   'HotpotQA': 0,
                     '2Wiki': 0, 'Musique': 0,  'Bamboogle': 0,  'Avg': 0},
    'Vanilla RAG':  {'NQ': 1.0, 'PopQA': 1.0, 'AmbigQA': 1.0, 'HotpotQA': 1.0,
                     '2Wiki': 1.0, 'Musique': 1.0, 'Bamboogle': 1.0, 'Avg': 1.0},
    'MAO-ARAG':     {'NQ': 1.0, 'PopQA': 1.0, 'AmbigQA': 1.0, 'HotpotQA': 3.536,
                     '2Wiki': 3.237, 'Musique': 3.413, 'Bamboogle': 3.104, 'Avg': 2.327},
}

# Table 7: Turns
BASELINE_TURNS = {
    'LLM w/o RAG':  {'NQ': 1.0, 'PopQA': 1.0, 'AmbigQA': 1.0, 'HotpotQA': 1.0,
                     '2Wiki': 1.0, 'Musique': 1.0, 'Bamboogle': 1.0, 'Avg': 1.0},
    'Vanilla RAG':  {'NQ': 1.0, 'PopQA': 1.0, 'AmbigQA': 1.0, 'HotpotQA': 1.0,
                     '2Wiki': 1.0, 'Musique': 1.0, 'Bamboogle': 1.0, 'Avg': 1.0},
    'MAO-ARAG':     {'NQ': 1.0, 'PopQA': 1.0, 'AmbigQA': 1.0, 'HotpotQA': 4.536,
                     '2Wiki': 4.237, 'Musique': 4.413, 'Bamboogle': 4.104, 'Avg': 2.899},
}

DATASETS = ['NQ', 'PopQA', 'AmbigQA', 'HotpotQA', '2Wiki', 'Musique', 'Bamboogle', 'Avg']


class TablesGenerator:
    """Generates comparison tables from collected eval metrics.

    Parameters
    ----------
    metrics_dir : str
        Directory containing *_eval_summary.json files.
    output_dir : str
        Where to write CSV / LaTeX / Markdown outputs.
    experiment_name : str
        Used to locate the eval summary file and name output files.
    condor_label : str
        Display name for the CONDOR method in tables (default 'CONDOR').
    """

    def __init__(
        self,
        metrics_dir: str,
        output_dir: str,
        experiment_name: str,
        condor_label: str = 'CONDOR',
    ):
        self.metrics_dir = metrics_dir
        self.output_dir = output_dir
        self.experiment_name = experiment_name
        self.condor_label = condor_label
        os.makedirs(output_dir, exist_ok=True)

        self._eval_summary: Dict[str, Any] = self._load_eval_summary()

    # ------------------------------------------------------------------
    # I/O helpers
    # ------------------------------------------------------------------

    def _load_eval_summary(self) -> Dict[str, Any]:
        path = os.path.join(self.metrics_dir, f'{self.experiment_name}_eval_summary.json')
        if os.path.exists(path):
            with open(path, 'r') as fh:
                return json.load(fh)
        return {}

    def _dataset_name_to_key(self, ds: str) -> str:
        return ds.lower().replace(' ', '_').replace('-', '_')

    def _get_condor_value(self, metric: str, dataset: str) -> Optional[float]:
        """Look up one CONDOR metric from the eval summary."""
        ds_key = self._dataset_name_to_key(dataset)
        if ds_key in self._eval_summary:
            return self._eval_summary[ds_key].get(metric)
        return None

    # ------------------------------------------------------------------
    # Table 1: F1 comparison
    # ------------------------------------------------------------------

    def generate_table1_f1(self) -> str:
        """Generate Table 1 (F1 %) as Markdown, CSV, and LaTeX."""
        rows = dict(BASELINE_F1)
        condor_row = {}
        for ds in DATASETS:
            if ds == 'Avg':
                vals = [v for v in condor_row.values() if v is not None]
                condor_row['Avg'] = sum(vals) / len(vals) if vals else None
            else:
                condor_row[ds] = self._get_condor_value('f1', ds)
                if condor_row[ds] is not None:
                    condor_row[ds] *= 100  # convert to %
        if any(v is not None for v in condor_row.values()):
            rows[self.condor_label] = condor_row

        return self._write_table(
            rows=rows,
            columns=DATASETS,
            title='Table 1: F1 performance (%) across datasets',
            filename='table1_f1',
            fmt='{:.2f}',
            bold_max=True,
            delta_ref='MAO-ARAG',
        )

    # ------------------------------------------------------------------
    # Table 5: Token cost
    # ------------------------------------------------------------------

    def generate_table5_token_cost(self) -> str:
        rows = dict(BASELINE_TOKEN_COST)
        condor_row = {}
        for ds in DATASETS:
            if ds == 'Avg':
                vals = [v for v in condor_row.values() if v is not None]
                condor_row['Avg'] = sum(vals) / len(vals) if vals else None
            else:
                val = self._get_condor_value('token_cost_milliusd_mean', ds)
                condor_row[ds] = val
        if any(v is not None for v in condor_row.values()):
            rows[self.condor_label] = condor_row

        return self._write_table(
            rows=rows,
            columns=DATASETS,
            title='Table 5: Average token cost per query (milli-USD)',
            filename='table5_token_cost',
            fmt='{:.3f}',
            bold_max=False,  # lower is better
            bold_min=True,
        )

    # ------------------------------------------------------------------
    # Table 6: Retrieval calls
    # ------------------------------------------------------------------

    def generate_table6_retrieval(self) -> str:
        rows = dict(BASELINE_RETRIEVAL)
        condor_row = {}
        for ds in DATASETS:
            if ds == 'Avg':
                vals = [v for v in condor_row.values() if v is not None]
                condor_row['Avg'] = sum(vals) / len(vals) if vals else None
            else:
                condor_row[ds] = self._get_condor_value('retrieval_calls_mean', ds)
        if any(v is not None for v in condor_row.values()):
            rows[self.condor_label] = condor_row

        return self._write_table(
            rows=rows,
            columns=DATASETS,
            title='Table 6: Average retrieval call times per query',
            filename='table6_retrieval',
            fmt='{:.3f}',
            bold_max=False,
            bold_min=True,
        )

    # ------------------------------------------------------------------
    # Table 7: Turns
    # ------------------------------------------------------------------

    def generate_table7_turns(self) -> str:
        rows = dict(BASELINE_TURNS)
        condor_row = {}
        for ds in DATASETS:
            if ds == 'Avg':
                vals = [v for v in condor_row.values() if v is not None]
                condor_row['Avg'] = sum(vals) / len(vals) if vals else None
            else:
                condor_row[ds] = self._get_condor_value('turns_mean', ds)
        if any(v is not None for v in condor_row.values()):
            rows[self.condor_label] = condor_row

        return self._write_table(
            rows=rows,
            columns=DATASETS,
            title='Table 7: Average number of total turns per query',
            filename='table7_turns',
            fmt='{:.3f}',
            bold_max=False,
            bold_min=True,
        )

    # ------------------------------------------------------------------
    # Table C: CONDOR mechanism breakdown (new)
    # ------------------------------------------------------------------

    def generate_table_mechanism_breakdown(self) -> str:
        from condor.mechanism_router import MECHANISM_NAMES
        mech_names = {str(k): v for k, v in MECHANISM_NAMES.items()}

        rows: Dict[str, Dict] = {}
        for ds_key, summary in self._eval_summary.items():
            row = {}
            for m_id in range(5):
                f1_key = f'mech_{m_id}_f1'
                cnt_key = f'mech_{m_id}_count'
                if f1_key in summary:
                    m_name = mech_names.get(str(m_id), f'm{m_id}')
                    row[f'{m_name} F1'] = summary[f1_key] * 100
                    row[f'{m_name} N'] = summary.get(cnt_key, '')
            if row:
                rows[ds_key.upper()] = row

        if not rows:
            return ''

        cols = list(next(iter(rows.values())).keys())
        return self._write_table(
            rows=rows,
            columns=cols,
            title='Table C: CONDOR per-mechanism F1 and query count',
            filename='tableC_mechanism_breakdown',
            fmt='{:.2f}',
            bold_max=True,
        )

    # ------------------------------------------------------------------
    # Table D: CONDOR Pareto summary (new)
    # ------------------------------------------------------------------

    def generate_table_pareto_summary(self) -> str:
        """Single-row CONDOR summary alongside baseline reference methods."""
        methods: Dict[str, Dict] = {
            'MAO-ARAG': {
                'F1 (%)': BASELINE_F1['MAO-ARAG']['Avg'],
                'Token Cost (mUSD)': BASELINE_TOKEN_COST['MAO-ARAG']['Avg'],
                'Retrieval Calls': BASELINE_RETRIEVAL['MAO-ARAG']['Avg'],
                'Turns': BASELINE_TURNS['MAO-ARAG']['Avg'],
            },
        }
        if self._eval_summary:
            all_f1, all_tc, all_ret, all_turn = [], [], [], []
            for summary in self._eval_summary.values():
                if 'f1' in summary:
                    all_f1.append(summary['f1'] * 100)
                if 'token_cost_milliusd_mean' in summary:
                    all_tc.append(summary['token_cost_milliusd_mean'])
                if 'retrieval_calls_mean' in summary:
                    all_ret.append(summary['retrieval_calls_mean'])
                if 'turns_mean' in summary:
                    all_turn.append(summary['turns_mean'])
            if all_f1:
                methods[self.condor_label] = {
                    'F1 (%)': sum(all_f1) / len(all_f1),
                    'Token Cost (mUSD)': sum(all_tc) / len(all_tc) if all_tc else None,
                    'Retrieval Calls': sum(all_ret) / len(all_ret) if all_ret else None,
                    'Turns': sum(all_turn) / len(all_turn) if all_turn else None,
                }

        cols = ['F1 (%)', 'Token Cost (mUSD)', 'Retrieval Calls', 'Turns']
        return self._write_table(
            rows=methods,
            columns=cols,
            title='Table D: Pareto summary — CONDOR vs baselines',
            filename='tableD_pareto_summary',
            fmt='{:.3f}',
            bold_max=True,
        )

    # ------------------------------------------------------------------
    # Generate all tables
    # ------------------------------------------------------------------

    def generate_all(self) -> None:
        self.generate_table1_f1()
        self.generate_table5_token_cost()
        self.generate_table6_retrieval()
        self.generate_table7_turns()
        self.generate_table_mechanism_breakdown()
        self.generate_table_pareto_summary()
        print(f'[TablesGenerator] All tables written to {self.output_dir}')

    # ------------------------------------------------------------------
    # Core table writer
    # ------------------------------------------------------------------

    def _write_table(
        self,
        rows: Dict[str, Dict],
        columns: List[str],
        title: str,
        filename: str,
        fmt: str = '{:.3f}',
        bold_max: bool = False,
        bold_min: bool = False,
        delta_ref: Optional[str] = None,
    ) -> str:
        """Write CSV, LaTeX, and Markdown versions of a table."""

        # ── determine per-column best value for bolding ──────────────
        col_max = {}
        col_min = {}
        for col in columns:
            vals = [v for row in rows.values()
                    if (v := row.get(col)) is not None and isinstance(v, (int, float))]
            if vals:
                col_max[col] = max(vals)
                col_min[col] = min(vals)

        def _fmt(val, col):
            if val is None:
                return '-'
            if not isinstance(val, (int, float)):
                return str(val)
            s = fmt.format(val)
            if bold_max and col in col_max and abs(val - col_max[col]) < 1e-9:
                s = f'**{s}**'
            elif bold_min and col in col_min and abs(val - col_min[col]) < 1e-9:
                s = f'**{s}**'
            return s

        # ── CSV ──────────────────────────────────────────────────────
        csv_path = os.path.join(self.output_dir, f'{filename}.csv')
        with open(csv_path, 'w', newline='', encoding='utf-8') as fh:
            writer = csv.DictWriter(fh, fieldnames=['Method'] + columns)
            writer.writeheader()
            for method, row in rows.items():
                csv_row = {'Method': method}
                csv_row.update({c: (fmt.format(row[c]) if isinstance(row.get(c), (int, float)) else (row.get(c) or ''))
                                for c in columns})
                writer.writerow(csv_row)

        # ── Markdown ─────────────────────────────────────────────────
        md_lines = [f'### {title}\n']
        header = '| Method | ' + ' | '.join(columns) + ' |'
        sep = '|---' * (len(columns) + 1) + '|'
        md_lines += [header, sep]
        for method, row in rows.items():
            cells = [_fmt(row.get(c), c) for c in columns]
            md_lines.append(f'| {method} | ' + ' | '.join(cells) + ' |')

        md_path = os.path.join(self.output_dir, f'{filename}.md')
        with open(md_path, 'w', encoding='utf-8') as fh:
            fh.write('\n'.join(md_lines) + '\n')

        # ── LaTeX ────────────────────────────────────────────────────
        col_spec = 'l' + 'c' * len(columns)
        latex_lines = [
            r'\begin{table}[t]',
            r'\centering',
            f'\\caption{{{title}}}',
            f'\\begin{{tabular}}{{{col_spec}}}',
            r'\toprule',
            'Methods & ' + ' & '.join(columns) + r' \\',
            r'\midrule',
        ]
        for method, row in rows.items():
            cells = []
            for c in columns:
                val = row.get(c)
                if val is None:
                    cells.append('-')
                elif not isinstance(val, (int, float)):
                    cells.append(str(val))
                else:
                    s = fmt.format(val)
                    if bold_max and c in col_max and abs(val - col_max[c]) < 1e-9:
                        s = f'\\textbf{{{s}}}'
                    elif bold_min and c in col_min and abs(val - col_min[c]) < 1e-9:
                        s = f'\\textbf{{{s}}}'
                    cells.append(s)
            latex_lines.append(f'{method} & ' + ' & '.join(cells) + r' \\')
        latex_lines += [
            r'\bottomrule',
            r'\end{tabular}',
            r'\end{table}',
        ]
        latex_path = os.path.join(self.output_dir, f'{filename}.tex')
        with open(latex_path, 'w', encoding='utf-8') as fh:
            fh.write('\n'.join(latex_lines) + '\n')

        return md_path
