#!/usr/bin/env python3
"""
Example Building Geometry for AAMKS-PINN Integration
Demonstrates apartment building setup similar to Grunnesjö thesis scenarios
"""

import os
import json
from collections import OrderedDict

class ExampleBuildingGeometry:
    """
    Creates a simple residential apartment building geometry for fire risk assessment.
    Based on extended travel distance scenarios from Grunnesjö 2014 thesis.
    """

    def __init__(self, project_path):
        self.project_path = project_path
        self.geometry = OrderedDict()
        self.fire_scenarios = []

    def create_apartment_building(self):
        """Create a 3-story apartment building with extended travel distances"""

        # Building parameters
        floor_height = 300  # cm
        corridor_width = 150  # cm
        room_depth = 400  # cm
        room_width = 350  # cm

        compartments = []
        doors = []
        stairs = []

        for floor in range(3):
            z_base = floor * floor_height

            # Create corridor for this floor
            corridor_id = f"COR{floor+1}"
            compartments.append({
                'name': corridor_id,
                'type_sec': 'COR',
                'type_pri': 'COMPA',
                'floor': floor,
                'x0': 0,
                'y0': 0,
                'width': 1000,  # 10m long corridor
                'depth': corridor_width,
                'height': floor_height,
                'z0': z_base,
                'fire_model_ignore': 0,
                'heat_detectors': 1,
                'smoke_detectors': 1,
                'sprinklers': 0
            })

            # Create apartments along corridor
            for apt in range(4):
                apt_id = f"APT{floor+1}{apt+1}"
                x_pos = 200 + apt * 200

                compartments.append({
                    'name': apt_id,
                    'type_sec': 'ROOM',
                    'type_pri': 'COMPA',
                    'floor': floor,
                    'x0': x_pos,
                    'y0': corridor_width,
                    'width': room_width,
                    'depth': room_depth,
                    'height': floor_height,
                    'z0': z_base,
                    'fire_model_ignore': 0,
                    'heat_detectors': 1,
                    'smoke_detectors': 1,
                    'sprinklers': 0
                })

                # Create door from apartment to corridor
                door_id = f"DOOR{floor+1}{apt+1}"
                doors.append({
                    'name': door_id,
                    'type_sec': 'DOOR',
                    'type_tri': 'DOOR',
                    'vent_from_name': apt_id,
                    'vent_to_name': corridor_id,
                    'width': 90,  # cm
                    'height': 200,  # cm
                    'sill': 0,
                    'face': 'FRONT',
                    'face_offset': room_width // 2,
                    'cfast_width': 90
                })

            # Create staircase connection
            if floor < 2:  # Connect to floor above
                stair_id = f"STAI{floor+1}{floor+2}"
                stairs.append({
                    'name': stair_id,
                    'type_sec': 'STAI',
                    'type_pri': 'COMPA',
                    'floor': floor,
                    'x0': 1050,
                    'y0': 0,
                    'width': 200,
                    'depth': 300,
                    'height': floor_height * 2,  # Connects two floors
                    'z0': z_base,
                    'fire_model_ignore': 0,
                    'heat_detectors': 1,
                    'smoke_detectors': 1,
                    'sprinklers': 0
                })

                # Door from corridor to staircase
                stair_door_id = f"SDOOR{floor+1}"
                doors.append({
                    'name': stair_door_id,
                    'type_sec': 'DOOR',
                    'type_tri': 'DOOR',
                    'vent_from_name': corridor_id,
                    'vent_to_name': stair_id,
                    'width': 120,
                    'height': 200,
                    'sill': 0,
                    'face': 'RIGHT',
                    'face_offset': 100,
                    'cfast_width': 120
                })

        # Ground floor exit door
        exit_door = {
            'name': 'EXIT1',
            'type_sec': 'DOOR',
            'type_tri': 'DOOR',
            'vent_from_name': 'COR1',
            'vent_to_name': 'OUTSIDE',
            'width': 150,
            'height': 200,
            'sill': 0,
            'face': 'FRONT',
            'face_offset': 50,
            'cfast_width': 150,
            'terminal_door': 1,
            'exit_weight': 1
        }
        doors.append(exit_door)

        self.geometry = {
            'compartments': compartments,
            'doors': doors,
            'stairs': stairs
        }

        return self.geometry

    def create_fire_scenarios(self):
        """Create probabilistic fire scenarios for Monte Carlo analysis"""

        scenarios = [
            {
                'name': 'apartment_bedroom_fire',
                'description': 'Fire starting in apartment bedroom',
                'fire_locations': ['APT11', 'APT12', 'APT21', 'APT22', 'APT31', 'APT32'],
                'fire_growth_rate': {'min': 0.012, 'mode': 0.024, 'max': 0.047},  # kW/s²
                'peak_hrr': {'min': 1000, 'mode': 2500, 'max': 5000},  # kW
                'door_open_probability': 0.7,
                'detection_time': {'mean': 120, 'sd': 30}  # seconds
            },
            {
                'name': 'corridor_fire',
                'description': 'Fire in corridor (electrical/trash)',
                'fire_locations': ['COR1', 'COR2', 'COR3'],
                'fire_growth_rate': {'min': 0.024, 'mode': 0.035, 'max': 0.065},
                'peak_hrr': {'min': 500, 'mode': 1500, 'max': 3000},
                'door_open_probability': 0.9,  # Doors likely open in corridor fire
                'detection_time': {'mean': 60, 'sd': 20}
            },
            {
                'name': 'staircase_fire',
                'description': 'Fire in staircase (storage/electrical)',
                'fire_locations': ['STAI12', 'STAI23'],
                'fire_growth_rate': {'min': 0.018, 'mode': 0.030, 'max': 0.055},
                'peak_hrr': {'min': 800, 'mode': 2000, 'max': 4000},
                'door_open_probability': 0.5,
                'detection_time': {'mean': 90, 'sd': 25}
            }
        ]

        self.fire_scenarios = scenarios
        return scenarios

    def create_conf_json(self):
        """Create AAMKS configuration file for this building"""

        conf = {
            'project_id': 1,
            'scenario_id': 1,
            'number_of_simulations': 1000,
            'simulation_time': 600,

            # Fire model selection
            'fire_model': 'PINN',  # Will use PINN instead of CFAST

            # Building configuration
            'building_category': 'residential',
            'floors': 3,
            'max_occupants': 24,  # 6 people per floor

            # Fire parameters
            'fire_starts_in_a_room': 0.8,  # 80% probability fire starts in room vs corridor
            'fuel': 'WOOD',  # Typical residential furnishing

            # Material properties
            'material_ceiling': {'type': 'gypsum', 'thickness': 0.015},
            'material_wall': {'type': 'gypsum', 'thickness': 0.015},
            'material_floor': {'type': 'concrete', 'thickness': 0.10},

            # Detection systems
            'heat_detectors': {
                'mean': 68,  # °C
                'sd': 5,
                'RTI': 50,
                'not_broken': 0.95
            },
            'smoke_detectors': {
                'mean': 0.1,  # m⁻¹ optical density
                'sd': 0.02,
                'not_broken': 0.98
            },

            # Evacuation parameters
            'pre_evac': {
                'mean': 120,  # seconds
                'sd': 60,
                '1st': 30,
                '99th': 300
            },

            # Door/window probabilities
            'vents_open': {
                'DOOR': 0.7,
                'DCLOSER': 0.4,
                'DELECTR': 0.2,
                'VVENT': 0.3
            },

            # Fire load densities (MJ/m²)
            'fire_load': {
                'room': {'mean': 500, 'sd': 200, '1st': 200, '99th': 1000},
                'non_room': {'mean': 100, 'sd': 50, '1st': 50, '99th': 300}
            },

            # Heat release rate per unit area
            'hrrpua': {'min': 150, 'mode': 250, 'max': 500},  # kW/m²

            # Fire growth rate
            'hrr_alpha': {'min': 0.012, 'mode': 0.024, 'max': 0.047},  # kW/s²

            # Outdoor conditions
            'outdoor_temperature': {'mean': 20, 'sd': 10},
            'indoor_temperature': {'mean': 22, 'sd': 3},
            'pressure': {'mean': 101325, 'sd': 1000},
            'humidity': {'mean': 50, 'sd': 20}
        }

        return conf

    def save_to_project(self):
        """Save geometry and configuration to AAMKS project directory"""

        # Create project directory
        os.makedirs(self.project_path, exist_ok=True)
        os.makedirs(f"{self.project_path}/workers", exist_ok=True)

        # Save configuration
        conf = self.create_conf_json()
        with open(f"{self.project_path}/conf.json", 'w') as f:
            json.dump(conf, f, indent=2)

        # Save geometry (would normally be in SQLite database)
        geometry = self.create_apartment_building()
        with open(f"{self.project_path}/geometry.json", 'w') as f:
            json.dump(geometry, f, indent=2)

        # Save fire scenarios
        scenarios = self.create_fire_scenarios()
        with open(f"{self.project_path}/fire_scenarios.json", 'w') as f:
            json.dump(scenarios, f, indent=2)

        print(f"Example building geometry saved to: {self.project_path}")
        print(f"Configuration: {len(geometry['compartments'])} compartments, {len(geometry['doors'])} doors")
        print(f"Fire scenarios: {len(scenarios)} scenario types")

        return self.project_path

def create_pinn_integration_example():
    """
    Creates a complete example showing how PINN fire modeling integrates with AAMKS
    """

    pinn_example = '''
# PINN Integration Example
# This shows how our PINN fire solver would replace CFAST in AAMKS

def run_pinn_fire_simulation(cfast_input_file, output_dir):
    """
    PINN-based fire simulation that replaces CFAST binary execution
    """
    from fire_pinn_framework.core.neural_network import FirePINN
    from fire_pinn_framework.utils.cfast_parser import CFASTInputParser

    # Parse CFAST input file (same format as before)
    parser = CFASTInputParser(cfast_input_file)
    geometry = parser.get_geometry()
    fire_params = parser.get_fire_parameters()
    materials = parser.get_materials()

    # Initialize PINN fire model
    fire_model = FirePINN(
        geometry=geometry,
        materials=materials,
        time_steps=600,  # 10 minute simulation
        dt=1.0           # 1 second time step
    )

    # Run physics-informed neural network simulation
    # This replaces the 10-300 second CFAST simulation with 0.01-0.3 second PINN
    results = fire_model.simulate(
        fire_source=fire_params,
        boundary_conditions=parser.get_boundary_conditions()
    )

    # Generate CFAST-compatible output files
    output_writer = CFASTOutputWriter(output_dir)
    output_writer.write_temperature_file(results.temperature)
    output_writer.write_smoke_file(results.smoke_density)
    output_writer.write_species_file(results.co_concentration)

    # Return success status (same interface as CFAST)
    return 0  # Success

# Integration in AAMKS worker.py:
def run_cfast_simulations(self, version='pinn', attempt=0):
    if self.project_conf['fire_model'] == 'PINN':
        # Use PINN instead of CFAST binary
        exit_code = run_pinn_fire_simulation("cfast.in", self.working_dir)
        if exit_code == 0:
            return True
    else:
        # Original CFAST execution
        p = run([f"{os.environ['AAMKS_PATH']}/fire/{cfast_file}", "cfast.in"],
                timeout=600, capture_output=True, text=True)
        return p.returncode == 0
'''

    return pinn_example

if __name__ == "__main__":
    # Create example building for AAMKS-PINN integration testing
    project_dir = "/data/data/com.termux/files/home/example_aamks_project"

    building = ExampleBuildingGeometry(project_dir)
    building.save_to_project()

    # Create PINN integration example
    pinn_code = create_pinn_integration_example()
    with open(f"{project_dir}/pinn_integration_example.py", 'w') as f:
        f.write(pinn_code)

    print("\n" + "="*60)
    print("AAMKS-PINN INTEGRATION EXAMPLE CREATED")
    print("="*60)
    print(f"Project directory: {project_dir}")
    print("\nBuilding configuration:")
    print("- 3-story apartment building")
    print("- 12 apartments (4 per floor)")
    print("- Central corridor with staircase")
    print("- Extended travel distances (Grunnesjö-style)")
    print("\nFire scenarios:")
    print("- Apartment fires (bedroom/living room)")
    print("- Corridor fires (electrical/trash)")
    print("- Staircase fires (storage areas)")
    print("\nNext steps:")
    print("1. Implement PINN fire solver in fire_pinn_framework/")
    print("2. Create CFAST input/output parser")
    print("3. Replace CFAST binary call in worker.py")
    print("4. Run 1000+ Monte Carlo simulations")
    print("5. Compare risk assessment results")