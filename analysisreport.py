import os
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd
# Changed import to get analyze_dose_accuracy from dose.py instead of dosimetry
from dose import analyze_dose_accuracy
from logging import log_info, log_error


class AnalysisReport:
    """Class for generating comprehensive analysis reports from simulation results."""

    def __init__(self, results_dict, output_dir='reports'):
        """
        Initialize the report generator.

        Parameters:
            results_dict: Dictionary of simulation results
            output_dir: Directory to save reports
        """
        self.results = results_dict
        self.output_dir = output_dir
        self.dose_accuracy_analysis = {}  # Added to store dose accuracy analysis results

        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            log_info(f"Created report directory: {output_dir}")

    def analyze_dose_accuracy(self, energy_kev, channel_diameter, experimental_data=None):
        """
        Analyze the accuracy of different dose calculation methods for a specific configuration.

        Args:
            energy_kev: Energy in keV
            channel_diameter: Channel diameter in cm
            experimental_data: Optional experimental data for comparison

        Returns:
            dict: Dose accuracy analysis
        """
        # Get results for this configuration
        result_key = f"{energy_kev}_{channel_diameter}"
        if result_key not in self.results:
            print(f"No results found for energy={energy_kev} keV, diameter={channel_diameter} cm")
            return None

        results = self.results[result_key]

        # Perform dose accuracy analysis
        accuracy_analysis = analyze_dose_accuracy(results, energy_kev, experimental_data)

        # Store analysis in results
        self.dose_accuracy_analysis[result_key] = accuracy_analysis

        return accuracy_analysis
    
    def generate_dose_accuracy_report(self):
        """
        Generate a comprehensive report on dose calculation accuracy across all simulations.

        Returns:
            dict: Dose accuracy report
        """
        if not self.dose_accuracy_analysis:
            print("No dose accuracy analyses available. Run analyze_dose_accuracy first.")
            return None

        # Summarize which method is most accurate for each configuration
        summary = {}
        for key, analysis in self.dose_accuracy_analysis.items():
            energy_kev, channel_diameter = key.split('_')
            energy_kev = float(energy_kev)
            channel_diameter = float(channel_diameter)

            most_accurate = analysis['most_accurate_method']

            if most_accurate not in summary:
                summary[most_accurate] = []

            summary[most_accurate].append({
                'energy_kev': energy_kev,
                'channel_diameter': channel_diameter,
                'comparison_metrics': analysis['comparison_results']
            })

        # Overall recommendation
        method_counts = {method: len(configs) for method, configs in summary.items()}
        overall_best = max(method_counts, key=method_counts.get)

        report = {
            'summary_by_method': summary,
            'method_counts': method_counts,
            'overall_recommendation': {
                'best_method': overall_best,
                'explanation': f"The {overall_best} method was most accurate in {method_counts[overall_best]} out of {sum(method_counts.values())} configurations."
            }
        }

        return report

    def determine_overall_best_dose_method(self, experimental_data=None):
        """
        Determine the overall best dose calculation method across all simulations.

        Args:
            experimental_data: Optional dictionary mapping config keys to experimental data
                Format: {'energy_channel': {'distance': [...], 'angle': [...], 'dose_rate': [...]}}

        Returns:
            dict: Overall analysis of the best dose calculation methods
        """
        # Initialize method accuracy tracking
        method_accuracy = {
            'heating': {'count': 0, 'avg_error': []},
            'kerma': {'count': 0, 'avg_error': []},
            'flux_to_dose': {'count': 0, 'avg_error': []}
        }

        # Track method accuracy by energy range
        energy_ranges = {
            'low': {'range': (0, 200), 'methods': {}},
            'medium': {'range': (200, 800), 'methods': {}},
            'high': {'range': (800, float('inf')), 'methods': {}}
        }

        # Analyze each configuration
        for result_key, results in self.results.items():
            energy_kev, channel_diameter = result_key.split('_')
            energy_kev = float(energy_kev)
            channel_diameter = float(channel_diameter)

            # Get experimental data for this configuration if available
            config_exp_data = None
            if experimental_data and result_key in experimental_data:
                config_exp_data = experimental_data[result_key]

            # Analyze dose accuracy
            accuracy_analysis = analyze_dose_accuracy(results, energy_kev, config_exp_data)

            # Store results
            self.dose_accuracy_analysis[result_key] = accuracy_analysis

            # Update method accuracy statistics
            most_accurate = accuracy_analysis['most_accurate_method']

            if most_accurate in method_accuracy:
                method_accuracy[most_accurate]['count'] += 1

                # If we have experimental data, track error metrics
                if config_exp_data and 'comparison_results' in accuracy_analysis:
                    comparison = accuracy_analysis['comparison_results']
                    if most_accurate in comparison and 'mape' in comparison[most_accurate]:
                        method_accuracy[most_accurate]['avg_error'].append(
                            comparison[most_accurate]['mape']
                        )

            # Update energy range statistics
            for range_name, range_info in energy_ranges.items():
                min_energy, max_energy = range_info['range']
                if min_energy <= energy_kev < max_energy:
                    if most_accurate not in range_info['methods']:
                        range_info['methods'][most_accurate] = 0
                    range_info['methods'][most_accurate] += 1
                    break

        # Calculate average errors where available
        for method in method_accuracy:
            if method_accuracy[method]['avg_error']:
                method_accuracy[method]['mean_error'] = np.mean(method_accuracy[method]['avg_error'])
                method_accuracy[method]['std_error'] = np.std(method_accuracy[method]['avg_error'])
            else:
                method_accuracy[method]['mean_error'] = None
                method_accuracy[method]['std_error'] = None

        # Determine overall best method
        best_method = max(method_accuracy, key=lambda x: method_accuracy[x]['count'])

        # Generate detailed report
        report = {
            'overall_best_method': best_method,
            'method_statistics': method_accuracy,
            'energy_range_analysis': energy_ranges,
            'recommendation': {
                'best_method': best_method,
                'explanation': (
                    f"The {best_method} method was most accurate in "
                    f"{method_accuracy[best_method]['count']} out of "
                    f"{sum(m['count'] for m in method_accuracy.values())} "
                    f"configurations."
                )
            }
        }

        # Add energy-specific recommendations
        energy_recommendations = []
        for range_name, range_info in energy_ranges.items():
            if range_info['methods']:
                best_for_range = max(range_info['methods'], key=range_info['methods'].get)
                min_energy, max_energy = range_info['range']

                if max_energy == float('inf'):
                    energy_str = f"> {min_energy} keV"
                else:
                    energy_str = f"{min_energy}-{max_energy} keV"

                energy_recommendations.append({
                    'energy_range': range_name,
                    'energy_range_str': energy_str,
                    'best_method': best_for_range,
                    'count': range_info['methods'][best_for_range],
                    'total': sum(range_info['methods'].values())
                })

        report['energy_recommendations'] = energy_recommendations

        return report

    def generate_pdf_report(self, filename='simulation_report.pdf'):
        """
        Generate a PDF report with all analysis results.

        Parameters:
            filename: Output PDF filename

        Returns:
            bool: True if report generation was successful, False otherwise
        """
        try:
            # Import PDF libraries
            from reportlab.lib.pagesizes import letter
            from reportlab.lib import colors
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.units import inch

            # Create PDF document
            output_path = os.path.join(self.output_dir, filename)
            doc = SimpleDocTemplate(output_path, pagesize=letter)
            styles = getSampleStyleSheet()

            # Create custom styles
            styles.add(ParagraphStyle(name='Title',
                                    parent=styles['Heading1'],
                                    fontSize=18,
                                    spaceAfter=12))

            styles.add(ParagraphStyle(name='Section',
                                    parent=styles['Heading2'],
                                    fontSize=14,
                                    spaceAfter=10))

            styles.add(ParagraphStyle(name='Subsection',
                                    parent=styles['Heading3'],
                                    fontSize=12,
                                    spaceAfter=8))

            # Build document content
            content = []

            # Title
            content.append(Paragraph("Gamma-Ray Streaming Simulation Report", styles['Title']))
            content.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
            content.append(Spacer(1, 0.2 * inch))

            # Summary section
            content.append(Paragraph("Executive Summary", styles['Section']))

            # Extract key results
            summary_data = self._extract_summary_data()

            summary_text = []
            summary_text.append(
                "This report summarizes the results of gamma-ray streaming simulations through channels in shielding materials.")

            if 'overall_risk' in summary_data:
                summary_text.append(f"Overall Streaming Risk: {summary_data['overall_risk']}")

            if 'recommendation' in summary_data:
                summary_text.append(f"Recommendation: {summary_data['recommendation']}")

            if 'best_shape' in summary_data and 'best_geometry' in summary_data:
                summary_text.append(
                    f"Optimal channel design: {summary_data['best_shape'].capitalize()} shape with {summary_data['best_geometry'][0]} cm diameter")

            for line in summary_text:
                content.append(Paragraph(line, styles['Normal']))
                content.append(Spacer(1, 0.1 * inch))

            # Add summary table
            if summary_data:
                summary_table_data = []

                # Table header
                summary_table_data.append(["Parameter", "Value"])

                # Table rows
                for key, value in summary_data.items():
                    if isinstance(value, (int, float, str)) and key not in ['overall_risk', 'recommendation']:
                        if isinstance(value, float):
                            value_str = f"{value:.4g}"
                        else:
                            value_str = str(value)
                        summary_table_data.append([key.replace('_', ' ').title(), value_str])

                # Create table
                if len(summary_table_data) > 1:
                    table = Table(summary_table_data, colWidths=[3 * inch, 2 * inch])
                    table.setStyle(TableStyle([
                        ('BACKGROUND', (0, 0), (1, 0), colors.lightgrey),
                        ('TEXTCOLOR', (0, 0), (1, 0), colors.black),
                        ('ALIGN', (0, 0), (1, 0), 'CENTER'),
                        ('FONTNAME', (0, 0), (1, 0), 'Helvetica-Bold'),
                        ('BOTTOMPADDING', (0, 0), (1, 0), 12),
                        ('GRID', (0, 0), (-1, -1), 1, colors.black),
                    ]))
                    content.append(table)
                    content.append(Spacer(1, 0.2 * inch))

            # Add key visualization images
            content.append(Paragraph("Key Visualizations", styles['Section']))

            # Check for visualization files and add them
            visualization_files = [
                ('streaming_summary.png', 'Streaming Effects Summary'),
                ('streaming_pathways.png', 'Radiation Streaming Pathways'),
                ('damage_analysis.png', 'Radiation Damage Analysis'),
                ('geometry_optimization.png', 'Channel Geometry Optimization')
            ]

            for viz_file, viz_title in visualization_files:
                viz_path = os.path.join(self.output_dir, '..', 'results', viz_file)
                if os.path.exists(viz_path):
                    content.append(Paragraph(viz_title, styles['Subsection']))
                    img = Image(viz_path, width=6 * inch, height=4.5 * inch)
                    content.append(img)
                    content.append(Spacer(1, 0.2 * inch))

            # Configuration Details
            content.append(Paragraph("Simulation Configuration", styles['Section']))

            # Extract configuration data from first simulation
            config_data = {}
            for key, value in self.results.items():
                if key.startswith('config_1'):
                    config_data = value
                    break

            if config_data:
                config_table_data = []

                # Table header
                config_table_data.append(["Parameter", "Value"])

                # Extract configuration parameters
                config_params = [
                    ('wall_thickness', 'Wall Thickness (cm)'),
                    ('channel_diameter', 'Channel Diameter (cm)'),
                    ('channel_shape', 'Channel Shape'),
                    ('material', 'Wall Material'),
                    ('energy', 'Source Energy (MeV)'),
                    ('source_strength', 'Source Strength'),
                    ('num_particles', 'Number of Particles Simulated')
                ]

                # Add rows to table
                for param_key, param_label in config_params:
                    if param_key in config_data:
                        value = config_data[param_key]
                        if isinstance(value, float):
                            value_str = f"{value:.4g}"
                        else:
                            value_str = str(value)
                        config_table_data.append([param_label, value_str])

                # Create table
                if len(config_table_data) > 1:
                    table = Table(config_table_data, colWidths=[3 * inch, 2 * inch])
                    table.setStyle(TableStyle([
                        ('BACKGROUND', (0, 0), (1, 0), colors.lightgrey),
                        ('TEXTCOLOR', (0, 0), (1, 0), colors.black),
                        ('ALIGN', (0, 0), (1, 0), 'CENTER'),
                        ('FONTNAME', (0, 0), (1, 0), 'Helvetica-Bold'),
                        ('BOTTOMPADDING', (0, 0), (1, 0), 12),
                        ('GRID', (0, 0), (-1, -1), 1, colors.black),
                    ]))
                    content.append(table)
                    content.append(Spacer(1, 0.2 * inch))

            # Results Analysis
            content.append(Paragraph("Results Analysis", styles['Section']))

            # Streaming Analysis
            if 'streaming_analysis' in self.results:
                content.append(Paragraph("Streaming Analysis", styles['Subsection']))

                streaming = self.results['streaming_analysis']
                streaming_text = []

                for key, value in streaming.items():
                    if isinstance(value, (int, float)):
                        if isinstance(value, float):
                            value_str = f"{value:.4g}"
                        else:
                            value_str = str(value)
                        streaming_text.append(f"{key.replace('_', ' ').title()}: {value_str}")

                for line in streaming_text:
                    content.append(Paragraph(line, styles['Normal']))

                content.append(Spacer(1, 0.1 * inch))

            # Damage Analysis
            if 'damage_analysis' in self.results:
                content.append(Paragraph("Radiation Damage Analysis", styles['Subsection']))

                damage = self.results['damage_analysis']
                damage_text = []

                for key, value in damage.items():
                    if isinstance(value, (int, float)):
                        if isinstance(value, float):
                            value_str = f"{value:.4g}"
                        else:
                            value_str = str(value)
                        damage_text.append(f"{key.replace('_', ' ').title()}: {value_str}")

                for line in damage_text:
                    content.append(Paragraph(line, styles['Normal']))

                content.append(Spacer(1, 0.1 * inch))

            # Comparative Results
            content.append(Paragraph("Comparative Results", styles['Section']))

            # Create comparison table of all configurations
            configs = [v for k, v in self.results.items() if k.startswith('config_')]

            if configs:
                # Parameters to compare
                compare_params = [
                    ('channel_diameter', 'Channel Diameter (cm)'),
                    ('channel_shape', 'Channel Shape'),
                    ('energy', 'Energy (MeV)'),
                    ('dose_rem_per_hr', 'Dose Rate (rem/hr)'),
                    ('streaming_factor', 'Streaming Factor'),
                    ('attenuation_factor', 'Attenuation Factor')
                ]

                # Create table header
                comparison_table_data = [["Configuration"] + [param[1] for param in compare_params]]

                # Add data rows
                for i, config in enumerate(configs):
                    row = [f"Config {i + 1}"]
                    for param_key, _ in compare_params:
                        if param_key in config:
                            value = config[param_key]
                            if isinstance(value, float):
                                value_str = f"{value:.4g}"
                            else:
                                value_str = str(value)
                        else:
                            value_str = "N/A"
                        row.append(value_str)
                    comparison_table_data.append(row)

                # Create table
                if len(comparison_table_data) > 1:
                    table = Table(comparison_table_data)
                    table.setStyle(TableStyle([
                        ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
                        ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
                        ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
                        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                        ('GRID', (0, 0), (-1, -1), 1, colors.black),
                    ]))
                    content.append(table)
                    content.append(Spacer(1, 0.2 * inch))

            # Recommendations
            content.append(Paragraph("Recommendations", styles['Section']))

            # Extract recommendations from summary results
            if 'summary_analysis' in self.results and 'recommendation' in self.results['summary_analysis']:
                recommendation = self.results['summary_analysis']['recommendation']
                content.append(Paragraph(recommendation, styles['Normal']))
            else:
                # Default recommendations if none exist
                recommendations = [
                    "1. For high-energy sources, use narrow channels with stepped geometries.",
                    "2. Regular inspections of channel walls are recommended to identify radiation damage.",
                    "3. Consider implementing additional shielding in high-dose areas identified in the streaming pathways analysis.",
                    "4. For critical applications, validation with physical measurements is recommended."
                ]

                for rec in recommendations:
                    content.append(Paragraph(rec, styles['Normal']))
                    content.append(Spacer(1, 0.1 * inch))

            # Build PDF
            doc.build(content)

            log_info(f"PDF report generated successfully: {output_path}")
            return True

        except Exception as e:
            log_error(f"Error generating PDF report: {e}")
            return False

    def generate_excel_report(self, filename='simulation_data.xlsx'):
        """
        Generate an Excel spreadsheet with all analysis results.

        Parameters:
            filename: Output Excel filename

        Returns:
            bool: True if report generation was successful, False otherwise
        """
        try:
            # Create Excel writer
            output_path = os.path.join(self.output_dir, filename)
            writer = pd.ExcelWriter(output_path, engine='xlsxwriter')

            # Sheet 1: Summary results
            summary_data = self._extract_summary_data()
            summary_df = pd.DataFrame({
                'Parameter': list(summary_data.keys()),
                'Value': list(summary_data.values())
            })
            summary_df.to_excel(writer, sheet_name='Summary', index=False)

            # Sheet 2: Configuration comparison
            configs = [v for k, v in self.results.items() if k.startswith('config_')]
            if configs:
                # Identify all possible parameters
                all_params = set()
                for config in configs:
                    all_params.update(config.keys())

                # Create DataFrame
                config_data = []
                for i, config in enumerate(configs):
                    row = {'Configuration': f"Config {i + 1}"}
                    for param in all_params:
                        if param in config:
                            row[param] = config[param]
                    config_data.append(row)

                config_df = pd.DataFrame(config_data)
                config_df.to_excel(writer, sheet_name='Configurations', index=False)

            # Sheet 3: Streaming results
            if 'streaming_analysis' in self.results:
                streaming = self.results['streaming_analysis']
                # Flatten nested dictionaries
                flat_streaming = {}
                for key, value in streaming.items():
                    if isinstance(value, dict):
                        for subkey, subvalue in value.items():
                            flat_streaming[f"{key}_{subkey}"] = subvalue
                    else:
                        flat_streaming[key] = value

                streaming_df = pd.DataFrame({
                    'Parameter': list(flat_streaming.keys()),
                    'Value': list(flat_streaming.values())
                })
                streaming_df.to_excel(writer, sheet_name='Streaming Analysis', index=False)

            # Sheet 4: Damage analysis
            if 'damage_analysis' in self.results:
                damage = self.results['damage_analysis']
                # Flatten nested dictionaries
                flat_damage = {}
                for key, value in damage.items():
                    if isinstance(value, dict):
                        for subkey, subvalue in value.items():
                            flat_damage[f"{key}_{subkey}"] = subvalue
                    else:
                        flat_damage[key] = value

                damage_df = pd.DataFrame({
                    'Parameter': list(flat_damage.keys()),
                    'Value': list(flat_damage.values())
                })
                damage_df.to_excel(writer, sheet_name='Damage Analysis', index=False)

            # Save Excel file
            writer.close()

            log_info(f"Excel report generated successfully: {output_path}")
            return True

        except Exception as e:
            log_error(f"Error generating Excel report: {e}")
            return False

    def generate_html_report(self, filename='simulation_report.html'):
        """
        Generate an HTML report with all analysis results.

        Parameters:
            filename: Output HTML filename

        Returns:
            bool: True if report generation was successful, False otherwise
        """
        try:
            # Create HTML template
            output_path = os.path.join(self.output_dir, filename)

            # Start building HTML content
            html_content = f"""
                    <!DOCTYPE html>
                    <html lang="en">
                    <head>
                        <meta charset="UTF-8">
                        <meta name="viewport" content="width=device-width, initial-scale=1.0">
                        <title>Gamma-Ray Streaming Simulation Report</title>
                        <style>
                            body {{
                                font-family: Arial, sans-serif;
                                line-height: 1.6;
                                margin: 20px;
                                padding: 0;
                                color: #333;
                            }}
                            h1 {{
                                color: #2c3e50;
                                border-bottom: 2px solid #3498db;
                                padding-bottom: 10px;
                            }}
                            h2 {{
                                color: #2980b9;
                                margin-top: 30px;
                            }}
                            h3 {{
                                color: #3498db;
                            }}
                            table {{
                                border-collapse: collapse;
                                width: 100%;
                                margin: 20px 0;
                            }}
                            th, td {{
                                border: 1px solid #ddd;
                                padding: 8px;
                                text-align: left;
                            }}
                            th {{
                                background-color: #f2f2f2;
                                font-weight: bold;
                            }}
                            tr:nth-child(even) {{
                                background-color: #f9f9f9;
                            }}
                            img {{
                                max-width: 100%;
                                height: auto;
                                margin: 20px 0;
                                border: 1px solid #ddd;
                            }}
                            .summary-box {{
                                background-color: #f8f9fa;
                                border-left: 4px solid #3498db;
                                padding: 15px;
                                margin: 20px 0;
                            }}
                            .recommendation {{
                                background-color: #e8f4f8;
                                border-left: 4px solid #27ae60;
                                padding: 15px;
                                margin: 20px 0;
                            }}
                        </style>
                    </head>
                    <body>
                        <h1>Gamma-Ray Streaming Simulation Report</h1>
                        <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                    """

            # Extract summary data
            summary_data = self._extract_summary_data()

            # Executive Summary
            html_content += """
                        <h2>Executive Summary</h2>
                        <div class="summary-box">
                            <p>This report summarizes the results of gamma-ray streaming simulations through channels in shielding materials.</p>
                    """

            if 'overall_risk' in summary_data:
                html_content += f"<p><strong>Overall Streaming Risk:</strong> {summary_data['overall_risk']}</p>"

            if 'recommendation' in summary_data:
                html_content += f"<p><strong>Recommendation:</strong> {summary_data['recommendation']}</p>"

            if 'best_shape' in summary_data and 'best_geometry' in summary_data:
                html_content += f"<p><strong>Optimal channel design:</strong> {summary_data['best_shape'].capitalize()} shape with {summary_data['best_geometry'][0]} cm diameter</p>"

            html_content += """
                </div>
            """

            # Summary Table
            html_content += """
                <h2>Key Results</h2>
                <table>
                    <tr>
                        <th>Parameter</th>
                        <th>Value</th>
                    </tr>
            """

            for key, value in summary_data.items():
                if isinstance(value, (int, float, str)) and key not in ['overall_risk', 'recommendation']:
                    if isinstance(value, float):
                        value_str = f"{value:.4g}"
                    else:
                        value_str = str(value)
                    html_content += f"""
                        <tr>
                            <td>{key.replace('_', ' ').title()}</td>
                            <td>{value_str}</td>
                        </tr>
                        """

            html_content += """
                </table>
            """

            # Key Visualizations
            html_content += """
                <h2>Key Visualizations</h2>
            """

            # Check for visualization files and add them
            visualization_files = [
                ('streaming_summary.png', 'Streaming Effects Summary'),
                ('streaming_pathways.png', 'Radiation Streaming Pathways'),
                ('damage_analysis.png', 'Radiation Damage Analysis'),
                ('geometry_optimization.png', 'Channel Geometry Optimization')
            ]

            for viz_file, viz_title in visualization_files:
                viz_path = os.path.join(self.output_dir, '..', 'results', viz_file)
                if os.path.exists(viz_path):
                    # Create a relative path from the HTML file to the image
                    rel_path = os.path.relpath(viz_path, os.path.dirname(output_path))
                    html_content += f"""
                        <h3>{viz_title}</h3>
                        <img src="{rel_path}" alt="{viz_title}">
                        """

            # Configuration Details
            html_content += """
                <h2>Simulation Configuration</h2>
                <table>
                    <tr>
                        <th>Parameter</th>
                        <th>Value</th>
                    </tr>
            """

            # Extract configuration data from first simulation
            config_data = {}
            for key, value in self.results.items():
                if key.startswith('config_1'):
                    config_data = value
                    break

            if config_data:
                # Extract configuration parameters
                config_params = [
                    ('wall_thickness', 'Wall Thickness (cm)'),
                    ('channel_diameter', 'Channel Diameter (cm)'),
                    ('channel_shape', 'Channel Shape'),
                    ('material', 'Wall Material'),
                    ('energy', 'Source Energy (MeV)'),
                    ('source_strength', 'Source Strength'),
                    ('num_particles', 'Number of Particles Simulated')
                ]

                # Add rows to table
                for param_key, param_label in config_params:
                    if param_key in config_data:
                        value = config_data[param_key]
                        if isinstance(value, float):
                            value_str = f"{value:.4g}"
                        else:
                            value_str = str(value)
                        html_content += f"""
                            <tr>
                                <td>{param_label}</td>
                                <td>{value_str}</td>
                            </tr>
                            """

            html_content += """
                </table>
            """

            # Results Analysis
            html_content += """
                <h2>Results Analysis</h2>
            """

            # Streaming Analysis
            if 'streaming_analysis' in self.results:
                html_content += """
                        <h3>Streaming Analysis</h3>
                        <table>
                            <tr>
                                <th>Parameter</th>
                                <th>Value</th>
                            </tr>
                    """

                streaming = self.results['streaming_analysis']

                for key, value in streaming.items():
                    if isinstance(value, (int, float)):
                        if isinstance(value, float):
                            value_str = f"{value:.4g}"
                        else:
                            value_str = str(value)
                        html_content += f"""
                            <tr>
                                <td>{key.replace('_', ' ').title()}</td>
                                <td>{value_str}</td>
                            </tr>
                            """

                html_content += """
                        </table>
                    """

            # Damage Analysis
            if 'damage_analysis' in self.results:
                html_content += """
                        <h3>Radiation Damage Analysis</h3>
                        <table>
                            <tr>
                                <th>Parameter</th>
                                <th>Value</th>
                            </tr>
                    """

                damage = self.results['damage_analysis']

                for key, value in damage.items():
                    if isinstance(value, (int, float)):
                        if isinstance(value, float):
                            value_str = f"{value:.4g}"
                        else:
                            value_str = str(value)
                        html_content += f"""
                            <tr>
                                <td>{key.replace('_', ' ').title()}</td>
                                <td>{value_str}</td>
                            </tr>
                            """

                html_content += """
                        </table>
                    """

            # Comparative Results
            html_content += """
                <h2>Comparative Results</h2>
            """

            # Create comparison table of all configurations
            configs = [v for k, v in self.results.items() if k.startswith('config_')]

            if configs:
                # Parameters to compare
                compare_params = [
                    ('channel_diameter', 'Channel Diameter (cm)'),
                    ('channel_shape', 'Channel Shape'),
                    ('energy', 'Energy (MeV)'),
                    ('dose_rem_per_hr', 'Dose Rate (rem/hr)'),
                    ('streaming_factor', 'Streaming Factor'),
                    ('attenuation_factor', 'Attenuation Factor')
                ]

                html_content += """
                        <table>
                            <tr>
                                <th>Configuration</th>
                    """

                for _, param_label in compare_params:
                    html_content += f"<th>{param_label}</th>"

                html_content += """
                            </tr>
                    """

                # Add data rows
                for i, config in enumerate(configs):
                    html_content += f"""
                            <tr>
                                <td>Config {i + 1}</td>
                        """

                    for param_key, _ in compare_params:
                        if param_key in config:
                            value = config[param_key]
                            if isinstance(value, float):
                                value_str = f"{value:.4g}"
                            else:
                                value_str = str(value)
                        else:
                            value_str = "N/A"

                        html_content += f"<td>{value_str}</td>"

                    html_content += """
                            </tr>
                        """

                html_content += """
                        </table>
                    """

            # Recommendations
            html_content += """
                <h2>Recommendations</h2>
                <div class="recommendation">
            """

            # Extract recommendations from summary results
            if 'summary_analysis' in self.results and 'recommendation' in self.results['summary_analysis']:
                recommendation = self.results['summary_analysis']['recommendation']
                html_content += f"<p>{recommendation}</p>"
            else:
                # Default recommendations if none exist
                recommendations = [
                    "1. For high-energy sources, use narrow channels with stepped geometries.",
                    "2. Regular inspections of channel walls are recommended to identify radiation damage.",
                    "3. Consider implementing additional shielding in high-dose areas identified in the streaming pathways analysis.",
                    "4. For critical applications, validation with physical measurements is recommended."
                ]

                for rec in recommendations:
                    html_content += f"<p>{rec}</p>"

            html_content += """
                </div>
            """

            # Close HTML document
            html_content += """
            </body>
            </html>
            """

            # Write HTML to file
            with open(output_path, 'w') as f:
                f.write(html_content)

            log_info(f"HTML report generated successfully: {output_path}")
            return True

        except Exception as e:
            log_error(f"Error generating HTML report: {e}")
            return False

    def _extract_summary_data(self):
        """
        Extract key summary data from results for report generation.

        Returns:
            dict: Dictionary of key results
        """
        summary_data = {}

        # Extract streaming analysis results
        if 'streaming_analysis' in self.results:
            streaming = self.results['streaming_analysis']

            # Copy top-level numeric values
            for key, value in streaming.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    summary_data[key] = value

        # Extract damage analysis results
        if 'damage_analysis' in self.results:
            damage = self.results['damage_analysis']

            # Copy top-level numeric values
            for key, value in damage.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    summary_data[key] = value

        # Extract geometry optimization results
        if 'geometry_optimization' in self.results:
            geometry = self.results['geometry_optimization']

            # Copy key geometry parameters
            if 'best_geometry' in geometry:
                summary_data['best_geometry'] = geometry['best_geometry']

            if 'best_shape' in geometry:
                summary_data['best_shape'] = geometry['best_shape']

        # Extract summary analysis results
        if 'summary_analysis' in self.results:
            summary = self.results['summary_analysis']

            # Copy key summary parameters
            for key in ['overall_risk', 'recommendation', 'optimal_diameter', 'optimal_energy']:
                if key in summary:
                    summary_data[key] = summary[key]

        return summary_data


def generate_report(results_dict, output_dir='reports', formats=None):
    """
    Generate analysis reports in specified formats.

    Parameters:
        results_dict: Dictionary of simulation results
        output_dir: Directory to save reports
        formats: List of report formats to generate ('pdf', 'excel', 'html')

    Returns:
        dict: Dictionary of generated report paths
    """
    if formats is None:
        formats = ['html', 'excel']  # Default formats

    report_generator = AnalysisReport(results_dict, output_dir)
    generated_reports = {}

    try:
        if 'pdf' in formats:
            pdf_filename = 'simulation_report.pdf'
            if report_generator.generate_pdf_report(pdf_filename):
                generated_reports['pdf'] = os.path.join(output_dir, pdf_filename)

        if 'excel' in formats:
            excel_filename = 'simulation_data.xlsx'
            if report_generator.generate_excel_report(excel_filename):
                generated_reports['excel'] = os.path.join(output_dir, excel_filename)

        if 'html' in formats:
            html_filename = 'simulation_report.html'
            if report_generator.generate_html_report(html_filename):
                generated_reports['html'] = os.path.join(output_dir, html_filename)

        return generated_reports

    except Exception as e:
        log_error(f"Error generating reports: {e}")
        return generated_reports
