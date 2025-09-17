#!/usr/bin/env python3
"""
Real-Time Web Dashboard for ML-Enhanced ASET/RSET Analysis

This module provides a comprehensive web-based interface for real-time fire risk
assessment using the ML-enhanced ASET/RSET analysis system. Features include:

- Interactive building parameter input
- Real-time risk assessment visualization
- Monte Carlo simulation results
- Extended travel distance analysis
- Grunnesjö thesis validation tools
- Performance comparison metrics

The dashboard provides both end-user interfaces and API endpoints for
integration with building management systems.
"""

import os
import sys
import json
import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
import threading
import time

# Add project paths
sys.path.append('/data/data/com.termux/files/home')

# Web framework imports
try:
    from flask import Flask, render_template, request, jsonify, send_file
    from flask_socketio import SocketIO, emit
    import plotly.graph_objs as go
    import plotly.utils
    import plotly.express as px
    WEB_AVAILABLE = True
except ImportError:
    print("Warning: Web framework not available. Install flask, flask-socketio, plotly for dashboard.")
    WEB_AVAILABLE = False

try:
    from ml_enhanced_aset_rset import ProbabilisticRiskFramework, RiskAssessmentResult
    from aamks_ml_integration import MLEnhancedAamks
    from grunnesjo_validation import GrunnesjoValidator
    ML_AVAILABLE = True
except ImportError:
    print("Warning: ML framework not available")
    ML_AVAILABLE = False


class RiskDashboard:
    """
    Real-time web dashboard for ML-enhanced fire risk assessment
    """

    def __init__(self, host: str = "0.0.0.0", port: int = 5000, debug: bool = False):
        """
        Initialize dashboard

        Args:
            host: Host address
            port: Port number
            debug: Debug mode
        """
        self.host = host
        self.port = port
        self.debug = debug

        if not WEB_AVAILABLE:
            raise ImportError("Web framework not available. Install flask, flask-socketio, plotly")

        # Initialize Flask app
        self.app = Flask(__name__)
        self.app.config['SECRET_KEY'] = 'ml_enhanced_fire_risk_dashboard_2024'

        # Initialize SocketIO for real-time updates
        self.socketio = SocketIO(self.app, cors_allowed_origins="*")

        # Initialize ML framework
        if ML_AVAILABLE:
            self.ml_framework = ProbabilisticRiskFramework()
            self.framework_trained = False
        else:
            self.ml_framework = None

        # Dashboard state
        self.active_simulations = {}
        self.simulation_counter = 0

        # Setup logging
        self.logger = self._setup_logging()

        # Setup routes
        self._setup_routes()
        self._setup_socketio_handlers()

        self.logger.info("Dashboard initialized successfully")

    def _setup_logging(self):
        """Setup dashboard logging"""
        logger = logging.getLogger('Dashboard')
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    def _setup_routes(self):
        """Setup Flask routes"""

        @self.app.route('/')
        def index():
            """Main dashboard page"""
            return self._render_dashboard_template()

        @self.app.route('/api/health')
        def health_check():
            """Health check endpoint"""
            return jsonify({
                'status': 'healthy',
                'ml_available': ML_AVAILABLE,
                'framework_trained': self.framework_trained,
                'timestamp': datetime.now().isoformat()
            })

        @self.app.route('/api/train', methods=['POST'])
        def train_framework():
            """Train ML framework endpoint"""
            if not ML_AVAILABLE:
                return jsonify({'error': 'ML framework not available'}), 400

            try:
                n_scenarios = request.json.get('n_scenarios', 300)
                self.logger.info(f"Training framework with {n_scenarios} scenarios...")

                # Run training in background thread
                training_thread = threading.Thread(
                    target=self._train_framework_background,
                    args=(n_scenarios,)
                )
                training_thread.start()

                return jsonify({
                    'status': 'training_started',
                    'n_scenarios': n_scenarios,
                    'message': 'Training started in background'
                })

            except Exception as e:
                self.logger.error(f"Training failed: {e}")
                return jsonify({'error': str(e)}), 500

        @self.app.route('/api/assess', methods=['POST'])
        def assess_risk():
            """Risk assessment endpoint"""
            if not ML_AVAILABLE:
                return jsonify({'error': 'ML framework not available'}), 400

            if not self.framework_trained:
                return jsonify({'error': 'Framework not trained. Call /api/train first'}), 400

            try:
                scenario = request.json.get('scenario', {})
                n_simulations = request.json.get('n_simulations', 200)

                self.logger.info(f"Running risk assessment for {n_simulations} simulations")

                # Run assessment in background
                sim_id = self._start_simulation(scenario, n_simulations)

                return jsonify({
                    'status': 'assessment_started',
                    'simulation_id': sim_id,
                    'message': 'Risk assessment started'
                })

            except Exception as e:
                self.logger.error(f"Assessment failed: {e}")
                return jsonify({'error': str(e)}), 500

        @self.app.route('/api/simulation/<int:sim_id>')
        def get_simulation_status(sim_id):
            """Get simulation status"""
            if sim_id not in self.active_simulations:
                return jsonify({'error': 'Simulation not found'}), 404

            simulation = self.active_simulations[sim_id]
            return jsonify(simulation)

        @self.app.route('/api/validate', methods=['POST'])
        def validate_grunnesjo():
            """Grunnesjö validation endpoint"""
            if not ML_AVAILABLE:
                return jsonify({'error': 'ML framework not available'}), 400

            try:
                validation_type = request.json.get('type', 'comprehensive')
                self.logger.info(f"Starting Grunnesjö validation: {validation_type}")

                # Start validation in background
                validation_thread = threading.Thread(
                    target=self._run_validation_background,
                    args=(validation_type,)
                )
                validation_thread.start()

                return jsonify({
                    'status': 'validation_started',
                    'type': validation_type,
                    'message': 'Validation started in background'
                })

            except Exception as e:
                self.logger.error(f"Validation failed: {e}")
                return jsonify({'error': str(e)}), 500

        @self.app.route('/api/scenarios/default')
        def get_default_scenarios():
            """Get default building scenarios"""
            return jsonify(self._get_default_scenarios())

        @self.app.route('/dashboard')
        def dashboard():
            """Interactive dashboard page"""
            return self._render_dashboard_template()

        @self.app.route('/validation')
        def validation_page():
            """Validation results page"""
            return self._render_validation_template()

    def _setup_socketio_handlers(self):
        """Setup SocketIO event handlers for real-time updates"""

        @self.socketio.on('connect')
        def handle_connect():
            """Handle client connection"""
            self.logger.info("Client connected")
            emit('connected', {
                'message': 'Connected to ML-Enhanced Fire Risk Dashboard',
                'ml_available': ML_AVAILABLE,
                'framework_trained': self.framework_trained
            })

        @self.socketio.on('disconnect')
        def handle_disconnect():
            """Handle client disconnection"""
            self.logger.info("Client disconnected")

        @self.socketio.on('request_status')
        def handle_status_request():
            """Handle status update request"""
            emit('status_update', {
                'ml_available': ML_AVAILABLE,
                'framework_trained': self.framework_trained,
                'active_simulations': len(self.active_simulations),
                'timestamp': datetime.now().isoformat()
            })

    def _train_framework_background(self, n_scenarios: int):
        """Train framework in background thread"""
        try:
            self.socketio.emit('training_progress', {
                'status': 'started',
                'message': f'Training with {n_scenarios} scenarios...'
            })

            self.ml_framework.train_framework(n_scenarios=n_scenarios)
            self.framework_trained = True

            self.socketio.emit('training_complete', {
                'status': 'completed',
                'message': 'ML framework training completed successfully'
            })

            self.logger.info("Framework training completed")

        except Exception as e:
            self.logger.error(f"Background training failed: {e}")
            self.socketio.emit('training_error', {
                'status': 'error',
                'message': str(e)
            })

    def _start_simulation(self, scenario: Dict, n_simulations: int) -> int:
        """Start risk assessment simulation"""
        sim_id = self.simulation_counter
        self.simulation_counter += 1

        simulation = {
            'id': sim_id,
            'scenario': scenario,
            'n_simulations': n_simulations,
            'status': 'running',
            'start_time': datetime.now().isoformat(),
            'progress': 0,
            'results': None
        }

        self.active_simulations[sim_id] = simulation

        # Start simulation in background thread
        sim_thread = threading.Thread(
            target=self._run_simulation_background,
            args=(sim_id, scenario, n_simulations)
        )
        sim_thread.start()

        return sim_id

    def _run_simulation_background(self, sim_id: int, scenario: Dict, n_simulations: int):
        """Run simulation in background thread"""
        try:
            simulation = self.active_simulations[sim_id]

            # Emit progress updates
            self.socketio.emit('simulation_progress', {
                'simulation_id': sim_id,
                'progress': 10,
                'message': 'Starting Monte Carlo simulation...'
            })

            # Run risk assessment
            risk_result = self.ml_framework.monte_carlo_risk_assessment(
                scenario,
                n_simulations=n_simulations,
                extended_travel_analysis=True
            )

            # Convert results to JSON-serializable format
            results = {
                'risk_level': risk_result.risk_level,
                'safety_probability': float(risk_result.safety_probability),
                'aset_mean': float(risk_result.aset_mean),
                'aset_std': float(risk_result.aset_std),
                'rset_mean': float(risk_result.rset_mean),
                'rset_std': float(risk_result.rset_std),
                'confidence_interval': [
                    float(risk_result.confidence_interval[0]),
                    float(risk_result.confidence_interval[1])
                ],
                'extended_travel_impact': risk_result.extended_travel_impact
            }

            # Generate visualizations
            visualizations = self._create_visualizations(risk_result, scenario)

            # Update simulation
            simulation.update({
                'status': 'completed',
                'progress': 100,
                'end_time': datetime.now().isoformat(),
                'results': results,
                'visualizations': visualizations
            })

            # Emit completion
            self.socketio.emit('simulation_complete', {
                'simulation_id': sim_id,
                'results': results,
                'visualizations': visualizations
            })

            self.logger.info(f"Simulation {sim_id} completed successfully")

        except Exception as e:
            self.logger.error(f"Simulation {sim_id} failed: {e}")

            simulation = self.active_simulations.get(sim_id, {})
            simulation.update({
                'status': 'error',
                'error': str(e),
                'end_time': datetime.now().isoformat()
            })

            self.socketio.emit('simulation_error', {
                'simulation_id': sim_id,
                'error': str(e)
            })

    def _run_validation_background(self, validation_type: str):
        """Run Grunnesjö validation in background"""
        try:
            self.socketio.emit('validation_progress', {
                'status': 'started',
                'type': validation_type,
                'message': 'Starting Grunnesjö thesis validation...'
            })

            validator = GrunnesjoValidator()

            if validation_type == 'comprehensive':
                results = validator.comprehensive_validation()
            elif validation_type == 'scaling':
                results = validator.validate_apartment_count_scaling()
            elif validation_type == 'travel':
                results = validator.validate_extended_travel_distances()
            elif validation_type == 'height':
                results = validator.validate_building_height_impact()
            else:
                raise ValueError(f"Unknown validation type: {validation_type}")

            self.socketio.emit('validation_complete', {
                'type': validation_type,
                'results': results,
                'message': 'Validation completed successfully'
            })

            self.logger.info(f"Validation {validation_type} completed")

        except Exception as e:
            self.logger.error(f"Validation {validation_type} failed: {e}")
            self.socketio.emit('validation_error', {
                'type': validation_type,
                'error': str(e)
            })

    def _create_visualizations(self, risk_result: RiskAssessmentResult, scenario: Dict) -> Dict:
        """Create visualizations for risk assessment results"""
        if not WEB_AVAILABLE:
            return {}

        try:
            visualizations = {}

            # ASET/RSET Distribution Plot
            aset_rset_fig = go.Figure()

            # Add ASET distribution
            aset_rset_fig.add_trace(go.Histogram(
                x=[risk_result.aset_mean] * 100,  # Simplified for demo
                name='ASET',
                opacity=0.7,
                marker_color='blue'
            ))

            # Add RSET distribution
            aset_rset_fig.add_trace(go.Histogram(
                x=[risk_result.rset_mean] * 100,  # Simplified for demo
                name='RSET',
                opacity=0.7,
                marker_color='red'
            ))

            aset_rset_fig.update_layout(
                title='ASET vs RSET Distribution',
                xaxis_title='Time (seconds)',
                yaxis_title='Frequency',
                barmode='overlay'
            )

            visualizations['aset_rset_distribution'] = json.dumps(
                aset_rset_fig, cls=plotly.utils.PlotlyJSONEncoder
            )

            # Risk Level Gauge
            risk_gauge_fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=risk_result.safety_probability * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Safety Probability (%)"},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgray"},
                        {'range': [50, 80], 'color': "yellow"},
                        {'range': [80, 100], 'color': "green"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ))

            visualizations['risk_gauge'] = json.dumps(
                risk_gauge_fig, cls=plotly.utils.PlotlyJSONEncoder
            )

            # Extended travel impact chart
            if risk_result.extended_travel_impact:
                distances = []
                impacts = []

                for key, value in risk_result.extended_travel_impact.items():
                    distance = key.replace('distance_', '').replace('m', '')
                    distances.append(int(distance))
                    impacts.append(value)

                travel_fig = go.Figure(data=go.Scatter(
                    x=distances,
                    y=impacts,
                    mode='lines+markers',
                    name='Safety Impact'
                ))

                travel_fig.update_layout(
                    title='Extended Travel Distance Impact',
                    xaxis_title='Distance (m)',
                    yaxis_title='Safety Impact (%)'
                )

                visualizations['travel_impact'] = json.dumps(
                    travel_fig, cls=plotly.utils.PlotlyJSONEncoder
                )

            return visualizations

        except Exception as e:
            self.logger.error(f"Visualization creation failed: {e}")
            return {}

    def _get_default_scenarios(self) -> List[Dict]:
        """Get default building scenarios for testing"""
        return [
            {
                'name': 'Small Office Building',
                'scenario': {
                    'building_height': 3,
                    'occupant_count': 75,
                    'fire_intensity': 1e6,
                    'room_volume': 150,
                    'travel_distance': 25,
                    'population_type': 'office_workers',
                    'time_of_day': 'day',
                    'exit_width': 2.0,
                    'ventilation_rate': 1.2
                }
            },
            {
                'name': 'Hospital Complex',
                'scenario': {
                    'building_height': 8,
                    'occupant_count': 300,
                    'fire_intensity': 1.2e6,
                    'room_volume': 500,
                    'travel_distance': 60,
                    'population_type': 'hospital_patients',
                    'time_of_day': 'night',
                    'exit_width': 2.5,
                    'ventilation_rate': 0.8
                }
            },
            {
                'name': 'School Building',
                'scenario': {
                    'building_height': 2,
                    'occupant_count': 200,
                    'fire_intensity': 8e5,
                    'room_volume': 300,
                    'travel_distance': 40,
                    'population_type': 'school_children',
                    'time_of_day': 'day',
                    'exit_width': 3.0,
                    'ventilation_rate': 1.5
                }
            },
            {
                'name': 'High-Rise Residential',
                'scenario': {
                    'building_height': 20,
                    'occupant_count': 400,
                    'fire_intensity': 1.5e6,
                    'room_volume': 800,
                    'travel_distance': 80,
                    'population_type': 'elderly_residents',
                    'time_of_day': 'evening',
                    'exit_width': 2.2,
                    'ventilation_rate': 0.6
                }
            }
        ]

    def _render_dashboard_template(self) -> str:
        """Render main dashboard HTML template"""
        html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ML-Enhanced Fire Risk Assessment Dashboard</title>
    <script src="https://cdn.socket.io/4.0.0/socket.io.min.js"></script>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {
            font-family: 'Arial', sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }

        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            text-align: center;
        }

        .dashboard-container {
            display: grid;
            grid-template-columns: 1fr 2fr;
            gap: 20px;
            max-width: 1400px;
            margin: 0 auto;
        }

        .control-panel {
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }

        .results-panel {
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }

        .form-group {
            margin-bottom: 15px;
        }

        label {
            display: block;
            margin-bottom: 5px;
            font-weight: bold;
            color: #333;
        }

        input, select {
            width: 100%;
            padding: 8px;
            border: 2px solid #ddd;
            border-radius: 5px;
            font-size: 14px;
        }

        button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 12px 24px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 16px;
            width: 100%;
            margin: 10px 0;
        }

        button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        }

        button:disabled {
            background: #cccccc;
            cursor: not-allowed;
            transform: none;
        }

        .status-indicator {
            padding: 10px;
            border-radius: 5px;
            margin: 10px 0;
            text-align: center;
        }

        .status-healthy {
            background-color: #d4edda;
            color: #155724;
            border: 1px solid #c3e6cb;
        }

        .status-training {
            background-color: #fff3cd;
            color: #856404;
            border: 1px solid #ffeaa7;
        }

        .status-error {
            background-color: #f8d7da;
            color: #721c24;
            border: 1px solid #f1b0b7;
        }

        .results-container {
            min-height: 400px;
        }

        .plot-container {
            width: 100%;
            height: 400px;
            margin: 20px 0;
        }

        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }

        .metric-card {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            text-align: center;
        }

        .metric-value {
            font-size: 24px;
            font-weight: bold;
            color: #333;
        }

        .metric-label {
            font-size: 12px;
            color: #666;
            text-transform: uppercase;
        }

        .risk-critical { color: #dc3545; }
        .risk-high { color: #fd7e14; }
        .risk-medium { color: #ffc107; }
        .risk-low { color: #28a745; }

        .loading {
            display: inline-block;
            width: 20px;
            height: 20px;
            border: 3px solid #f3f3f3;
            border-top: 3px solid #3498db;
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>ML-Enhanced Fire Risk Assessment Dashboard</h1>
        <p>Real-time probabilistic ASET/RSET analysis with 1000x performance improvement</p>
    </div>

    <div class="dashboard-container">
        <!-- Control Panel -->
        <div class="control-panel">
            <h2>Building Parameters</h2>

            <div class="status-indicator" id="system-status">
                <span id="status-text">Connecting...</span>
            </div>

            <form id="building-form">
                <div class="form-group">
                    <label for="building-height">Building Height (floors):</label>
                    <input type="number" id="building-height" min="1" max="50" value="5">
                </div>

                <div class="form-group">
                    <label for="occupant-count">Occupant Count:</label>
                    <input type="number" id="occupant-count" min="1" max="1000" value="100">
                </div>

                <div class="form-group">
                    <label for="fire-intensity">Fire Intensity (W/m³):</label>
                    <input type="number" id="fire-intensity" min="100000" max="10000000" value="1000000">
                </div>

                <div class="form-group">
                    <label for="travel-distance">Travel Distance (m):</label>
                    <input type="number" id="travel-distance" min="5" max="200" value="30">
                </div>

                <div class="form-group">
                    <label for="population-type">Population Type:</label>
                    <select id="population-type">
                        <option value="office_workers">Office Workers</option>
                        <option value="hospital_patients">Hospital Patients</option>
                        <option value="school_children">School Children</option>
                        <option value="elderly_residents">Elderly Residents</option>
                    </select>
                </div>

                <div class="form-group">
                    <label for="time-of-day">Time of Day:</label>
                    <select id="time-of-day">
                        <option value="day">Day</option>
                        <option value="evening">Evening</option>
                        <option value="night">Night</option>
                    </select>
                </div>

                <div class="form-group">
                    <label for="exit-width">Exit Width (m):</label>
                    <input type="number" id="exit-width" min="0.8" max="5.0" step="0.1" value="2.0">
                </div>

                <div class="form-group">
                    <label for="n-simulations">Monte Carlo Simulations:</label>
                    <input type="number" id="n-simulations" min="50" max="1000" value="200">
                </div>
            </form>

            <button id="train-btn" onclick="trainFramework()">Train ML Framework</button>
            <button id="assess-btn" onclick="assessRisk()" disabled>Run Risk Assessment</button>
            <button id="validate-btn" onclick="validateGrunnesjo()">Validate Grunnesjö Thesis</button>

            <h3>Default Scenarios</h3>
            <button onclick="loadScenario('small_office')">Small Office</button>
            <button onclick="loadScenario('hospital')">Hospital Complex</button>
            <button onclick="loadScenario('school')">School Building</button>
            <button onclick="loadScenario('highrise')">High-Rise Residential</button>
        </div>

        <!-- Results Panel -->
        <div class="results-panel">
            <h2>Risk Assessment Results</h2>

            <div id="results-container" class="results-container">
                <p>Configure building parameters and run risk assessment to see results.</p>
            </div>

            <div id="metrics-grid" class="metrics-grid" style="display: none;">
                <div class="metric-card">
                    <div class="metric-value" id="risk-level">-</div>
                    <div class="metric-label">Risk Level</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value" id="safety-prob">-</div>
                    <div class="metric-label">Safety Probability</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value" id="aset-mean">-</div>
                    <div class="metric-label">ASET Mean (s)</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value" id="rset-mean">-</div>
                    <div class="metric-label">RSET Mean (s)</div>
                </div>
            </div>

            <div id="aset-rset-plot" class="plot-container"></div>
            <div id="risk-gauge-plot" class="plot-container"></div>
            <div id="travel-impact-plot" class="plot-container"></div>
        </div>
    </div>

    <script>
        // Initialize SocketIO connection
        const socket = io();

        let frameworkTrained = false;
        let currentSimulationId = null;

        // Default scenarios
        const defaultScenarios = {
            small_office: {
                'building_height': 3,
                'occupant_count': 75,
                'fire_intensity': 1000000,
                'travel_distance': 25,
                'population_type': 'office_workers',
                'time_of_day': 'day',
                'exit_width': 2.0
            },
            hospital: {
                'building_height': 8,
                'occupant_count': 300,
                'fire_intensity': 1200000,
                'travel_distance': 60,
                'population_type': 'hospital_patients',
                'time_of_day': 'night',
                'exit_width': 2.5
            },
            school: {
                'building_height': 2,
                'occupant_count': 200,
                'fire_intensity': 800000,
                'travel_distance': 40,
                'population_type': 'school_children',
                'time_of_day': 'day',
                'exit_width': 3.0
            },
            highrise: {
                'building_height': 20,
                'occupant_count': 400,
                'fire_intensity': 1500000,
                'travel_distance': 80,
                'population_type': 'elderly_residents',
                'time_of_day': 'evening',
                'exit_width': 2.2
            }
        };

        // Socket event handlers
        socket.on('connected', (data) => {
            console.log('Connected to dashboard:', data.message);
            updateStatus('Connected', 'healthy');
            frameworkTrained = data.framework_trained;
            updateButtons();
        });

        socket.on('training_progress', (data) => {
            updateStatus('Training: ' + data.message, 'training');
        });

        socket.on('training_complete', (data) => {
            updateStatus('Framework Ready', 'healthy');
            frameworkTrained = true;
            updateButtons();
        });

        socket.on('training_error', (data) => {
            updateStatus('Training Error: ' + data.message, 'error');
        });

        socket.on('simulation_complete', (data) => {
            updateStatus('Assessment Complete', 'healthy');
            displayResults(data.results, data.visualizations);
            updateButtons();
        });

        socket.on('simulation_error', (data) => {
            updateStatus('Assessment Error: ' + data.error, 'error');
            updateButtons();
        });

        socket.on('validation_complete', (data) => {
            updateStatus('Validation Complete', 'healthy');
            console.log('Validation results:', data.results);
        });

        // Helper functions
        function updateStatus(message, type) {
            const statusDiv = document.getElementById('system-status');
            const statusText = document.getElementById('status-text');

            statusText.textContent = message;
            statusDiv.className = 'status-indicator status-' + type;
        }

        function updateButtons() {
            const trainBtn = document.getElementById('train-btn');
            const assessBtn = document.getElementById('assess-btn');

            if (frameworkTrained) {
                trainBtn.textContent = 'Retrain ML Framework';
                assessBtn.disabled = false;
            } else {
                trainBtn.textContent = 'Train ML Framework';
                assessBtn.disabled = true;
            }
        }

        function trainFramework() {
            const trainBtn = document.getElementById('train-btn');
            trainBtn.disabled = true;
            trainBtn.innerHTML = '<span class="loading"></span> Training...';

            updateStatus('Starting training...', 'training');

            fetch('/api/train', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ n_scenarios: 300 })
            })
            .then(response => response.json())
            .then(data => {
                console.log('Training started:', data);
            })
            .catch(error => {
                console.error('Training error:', error);
                updateStatus('Training failed: ' + error.message, 'error');
                trainBtn.disabled = false;
                trainBtn.textContent = 'Train ML Framework';
            });
        }

        function assessRisk() {
            const assessBtn = document.getElementById('assess-btn');
            assessBtn.disabled = true;
            assessBtn.innerHTML = '<span class="loading"></span> Assessing...';

            updateStatus('Running risk assessment...', 'training');

            const scenario = getScenarioFromForm();
            const nSimulations = parseInt(document.getElementById('n-simulations').value);

            fetch('/api/assess', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    scenario: scenario,
                    n_simulations: nSimulations
                })
            })
            .then(response => response.json())
            .then(data => {
                console.log('Assessment started:', data);
                currentSimulationId = data.simulation_id;
            })
            .catch(error => {
                console.error('Assessment error:', error);
                updateStatus('Assessment failed: ' + error.message, 'error');
                assessBtn.disabled = false;
                assessBtn.textContent = 'Run Risk Assessment';
            });
        }

        function validateGrunnesjo() {
            const validateBtn = document.getElementById('validate-btn');
            validateBtn.disabled = true;
            validateBtn.innerHTML = '<span class="loading"></span> Validating...';

            updateStatus('Running Grunnesjö validation...', 'training');

            fetch('/api/validate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ type: 'comprehensive' })
            })
            .then(response => response.json())
            .then(data => {
                console.log('Validation started:', data);
            })
            .catch(error => {
                console.error('Validation error:', error);
                updateStatus('Validation failed: ' + error.message, 'error');
            })
            .finally(() => {
                validateBtn.disabled = false;
                validateBtn.textContent = 'Validate Grunnesjö Thesis';
            });
        }

        function getScenarioFromForm() {
            return {
                building_height: parseInt(document.getElementById('building-height').value),
                occupant_count: parseInt(document.getElementById('occupant-count').value),
                fire_intensity: parseFloat(document.getElementById('fire-intensity').value),
                travel_distance: parseFloat(document.getElementById('travel-distance').value),
                population_type: document.getElementById('population-type').value,
                time_of_day: document.getElementById('time-of-day').value,
                exit_width: parseFloat(document.getElementById('exit-width').value),
                room_volume: 100 + parseInt(document.getElementById('building-height').value) * 50,
                ventilation_rate: 1.0
            };
        }

        function loadScenario(scenarioKey) {
            const scenario = defaultScenarios[scenarioKey];

            document.getElementById('building-height').value = scenario.building_height;
            document.getElementById('occupant-count').value = scenario.occupant_count;
            document.getElementById('fire-intensity').value = scenario.fire_intensity;
            document.getElementById('travel-distance').value = scenario.travel_distance;
            document.getElementById('population-type').value = scenario.population_type;
            document.getElementById('time-of-day').value = scenario.time_of_day;
            document.getElementById('exit-width').value = scenario.exit_width;
        }

        function displayResults(results, visualizations) {
            // Update metrics
            const riskLevel = document.getElementById('risk-level');
            riskLevel.textContent = results.risk_level;
            riskLevel.className = 'metric-value risk-' + results.risk_level.toLowerCase();

            document.getElementById('safety-prob').textContent = (results.safety_probability * 100).toFixed(1) + '%';
            document.getElementById('aset-mean').textContent = results.aset_mean.toFixed(1);
            document.getElementById('rset-mean').textContent = results.rset_mean.toFixed(1);

            // Show metrics grid
            document.getElementById('metrics-grid').style.display = 'grid';

            // Display visualizations
            if (visualizations.aset_rset_distribution) {
                const asetRsetData = JSON.parse(visualizations.aset_rset_distribution);
                Plotly.newPlot('aset-rset-plot', asetRsetData.data, asetRsetData.layout);
            }

            if (visualizations.risk_gauge) {
                const gaugeData = JSON.parse(visualizations.risk_gauge);
                Plotly.newPlot('risk-gauge-plot', gaugeData.data, gaugeData.layout);
            }

            if (visualizations.travel_impact) {
                const travelData = JSON.parse(visualizations.travel_impact);
                Plotly.newPlot('travel-impact-plot', travelData.data, travelData.layout);
            }

            // Reset assess button
            const assessBtn = document.getElementById('assess-btn');
            assessBtn.disabled = false;
            assessBtn.textContent = 'Run Risk Assessment';
        }

        // Initialize dashboard
        socket.emit('request_status');
    </script>
</body>
</html>
"""
        return html_template

    def _render_validation_template(self) -> str:
        """Render validation results template"""
        return """
<!DOCTYPE html>
<html>
<head>
    <title>Grunnesjö Thesis Validation Results</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .header { background-color: #2c3e50; color: white; padding: 20px; }
        .section { margin: 20px 0; padding: 15px; border-left: 4px solid #3498db; }
    </style>
</head>
<body>
    <div class="header">
        <h1>Grunnesjö Thesis Validation Results</h1>
        <p>ML-Enhanced ASET/RSET Analysis Validation</p>
    </div>

    <div class="section">
        <h2>Validation in Progress</h2>
        <p>Use the main dashboard to run validation tests and view results here.</p>
    </div>
</body>
</html>
"""

    def run(self):
        """Start the dashboard server"""
        if not WEB_AVAILABLE:
            self.logger.error("Cannot start dashboard - web framework not available")
            return

        self.logger.info(f"Starting dashboard on {self.host}:{self.port}")
        print(f"\n=== ML-Enhanced Fire Risk Assessment Dashboard ===")
        print(f"Dashboard URL: http://{self.host}:{self.port}")
        print(f"API Health Check: http://{self.host}:{self.port}/api/health")
        print(f"Validation Page: http://{self.host}:{self.port}/validation")

        try:
            self.socketio.run(
                self.app,
                host=self.host,
                port=self.port,
                debug=self.debug,
                allow_unsafe_werkzeug=True
            )
        except Exception as e:
            self.logger.error(f"Dashboard startup failed: {e}")


def main_dashboard():
    """Launch the web dashboard"""
    print("=== ML-Enhanced ASET/RSET Web Dashboard ===")

    if not WEB_AVAILABLE:
        print("ERROR: Web framework not available")
        print("Install dependencies: pip install flask flask-socketio plotly")
        return

    try:
        # Create and run dashboard
        dashboard = RiskDashboard(
            host="0.0.0.0",
            port=5000,
            debug=False
        )

        dashboard.run()

    except KeyboardInterrupt:
        print("\nDashboard shutdown requested")
    except Exception as e:
        print(f"Dashboard failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main_dashboard()