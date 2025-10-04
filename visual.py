import requests
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.widgets as widgets
from matplotlib.gridspec import GridSpec
import io
from PIL import Image

# =========================
# Configuration
# =========================
# NOTE: This API key is used as a default.
# The user will enter their key in the GUI.
API_KEY = "v4Lf0qNUv8pxjgpaZmtSOju7yUSGgIogV67qn3zM"
BASE_URL = "https://api.nasa.gov/neo/rest/v1"
MAP_URL = "https://eoimages.gsfc.nasa.gov/images/imagerecords/73000/73909/world.topo.bathy.200412.3x5400x2700.jpg"

# =========================
# Data & Calculation Classes (from original script)
# =========================

class NasaNeoAPI:
    """Handles fetching and parsing NEO data from NASA."""
    def __init__(self, api_key):
        self.api_key = api_key

    def fetch_recent_neos(self, days=7):
        """Fetches all NEOs from the past few days."""
        end_date = np.datetime64('today')
        start_date = end_date - np.timedelta64(days, 'D')
        
        # Format dates for the API call
        start_str = np.datetime_as_string(start_date, unit='D')
        end_str = np.datetime_as_string(end_date, unit='D')

        url = f"{BASE_URL}/feed"
        params = {"start_date": start_str, "end_date": end_str, "api_key": self.api_key}
        
        try:
            r = requests.get(url, params=params)
            r.raise_for_status()
            data = r.json()
            return self._parse_neos(data)
        except requests.exceptions.RequestException as e:
            print(f"Error fetching data: {e}")
            # In a GUI, we should show this error to the user
            # For simplicity, we'll just return an empty list here.
            return []

    def _parse_neos(self, raw_data):
        """Extracts and returns a list of NEOs with relevant stats."""
        all_neos = []
        seen_ids = set()

        if "near_earth_objects" not in raw_data:
            return all_neos

        for date in raw_data["near_earth_objects"]:
            for neo in raw_data["near_earth_objects"][date]:
                if neo['id'] in seen_ids or not neo.get("close_approach_data") or not neo.get("estimated_diameter"):
                    continue
                
                velocity_kps = neo['close_approach_data'][0]['relative_velocity'].get('kilometers_per_second')
                if not velocity_kps:
                    continue

                diameter_km = (neo["estimated_diameter"]["kilometers"]["estimated_diameter_min"] + 
                               neo["estimated_diameter"]["kilometers"]["estimated_diameter_max"]) / 2

                all_neos.append({
                    "id": neo["id"],
                    "name": neo["name"],
                    "diameter_km": diameter_km,
                    "velocity_kps": float(velocity_kps),
                })
                seen_ids.add(neo['id'])
        
        return sorted(all_neos, key=lambda x: x["diameter_km"], reverse=True)


class ImpactCalculator:
    """Calculates the physical effects of an asteroid impact."""
    def _init_(self, neo, angle_degrees=45, target_density_kgm3=2500):
        self.ASTEROID_DENSITY_KGM3 = 3000
        self.neo = neo
        self.angle_rad = np.deg2rad(angle_degrees)
        self.target_density_kgm3 = target_density_kgm3
        
        diameter_m = self.neo["diameter_km"] * 1000
        velocity_mps = self.neo["velocity_kps"] * 1000
        mass_kg = (4/3) * np.pi * ((diameter_m / 2)**3) * self.ASTEROID_DENSITY_KGM3
        self.impact_energy_joules = 0.5 * mass_kg * (velocity_mps * np.sin(self.angle_rad))**2

    def get_effects(self):
        """Returns a dictionary of all calculated impact effects."""
        energy_megatons_tnt = self.impact_energy_joules / 4.184e15
        energy_kilotons_tnt = self.impact_energy_joules / 4.184e12

        GROUND_ENERGY_FRACTION = 0.20
        fragmented_energy_kilotons = energy_kilotons_tnt * GROUND_ENERGY_FRACTION
        fragmented_crater_km = 0.02 * (fragmented_energy_kilotons**0.294)
        
        V_SCALE_KPS = 12.0
        BETA_EXP = 0.44
        density_ratio = self.ASTEROID_DENSITY_KGM3 / self.target_density_kgm3
        velocity_ratio = self.neo["velocity_kps"] / V_SCALE_KPS
        angle_term = np.cbrt(np.sin(self.angle_rad))
        solid_body_crater_km = 1.161 * self.neo["diameter_km"] * np.cbrt(density_ratio) * (velocity_ratio**BETA_EXP) * angle_term

        fireball_radius_km = 0.03 * (energy_megatons_tnt**0.33)
        magnitude = 0.67 * np.log10(self.impact_energy_joules) - 5.87

        return {
            "energy_megatons_tnt": energy_megatons_tnt,
            "fragmented_crater_km": fragmented_crater_km,
            "solid_body_crater_km": solid_body_crater_km,
            "fireball_radius_km": fireball_radius_km,
            "earthquake_magnitude": magnitude,
        }

# =========================
# GUI Application Class
# =========================

class AppGUI:
    def __init__(self):
        self.neos = []
        self.map_image = None
        
        # --- Create Figure and Axes ---
        self.fig = plt.figure(figsize=(16, 8))
        self.fig.canvas.manager.set_window_title('Asteroid Impact Simulator')
        gs = GridSpec(1, 3, figure=self.fig)
        
        # Map Axe
        self.ax_map = self.fig.add_subplot(gs[0, :2]) # Map takes 2/3 of the space
        self.ax_map.set_xlabel("Longitude")
        self.ax_map.set_ylabel("Latitude")
        self.ax_map.set_title("Impact Visualization")

        # Control Panel Axe
        self.ax_controls = self.fig.add_subplot(gs[0, 2]) # Controls take 1/3
        self.ax_controls.axis('off') # Hide axes for controls

        # --- Draw Initial Map ---
        self._load_map_image()
        if self.map_image:
            self.ax_map.imshow(self.map_image, extent=[-180, 180, -90, 90])
        
        # --- Create Widgets ---
        self._create_widgets()
        # Connect the map click event
        self.fig.canvas.mpl_connect('button_press_event', self._on_map_click)
        # Use subplots_adjust for manual layout control to avoid the UserWarning
        self.fig.subplots_adjust(left=0.05, right=0.65, top=0.95, bottom=0.05)
        plt.show()

    def _load_map_image(self):
        print("Loading high-resolution map...")
        try:
            response = requests.get(MAP_URL)
            response.raise_for_status()
            self.map_image = Image.open(io.BytesIO(response.content))
            print("Map loaded successfully.")
        except requests.exceptions.RequestException as e:
            print(f"Could not download map image: {e}")
            
    def _on_map_click(self, event):
        # Check if the click was on the map axes and was a left-click
        if event.inaxes != self.ax_map or event.button != 1:
            return
        
        # Get coordinates from the click event
        lon, lat = event.xdata, event.ydata
        
        # Update the text boxes with the new coordinates
        self.lat_textbox.set_val(f"{lat:.4f}")
        self.lon_textbox.set_val(f"{lon:.4f}")
        
        # Add a visual marker for the selected point, removing the old one first
        if hasattr(self, 'click_marker') and self.click_marker in self.ax_map.lines:
            self.click_marker.remove()
            
        self.click_marker, = self.ax_map.plot(lon, lat, 'g+', markersize=10, label='impact_selection_marker')
        
        self.fig.canvas.draw_idle()

    def _create_widgets(self):
        # API Key
        self.ax_controls.text(0.05, 0.95, "1. NASA API Key", transform=self.ax_controls.transAxes, fontsize=10, fontweight='bold')
        ax_api_box = self.fig.add_axes([0.7, 0.90, 0.25, 0.04])
        self.api_textbox = widgets.TextBox(ax_api_box, '', initial="DEMO_KEY")

        # Fetch Button
        ax_fetch_btn = self.fig.add_axes([0.7, 0.84, 0.25, 0.04])
        self.fetch_button = widgets.Button(ax_fetch_btn, 'Fetch Recent Asteroids')
        self.fetch_button.on_clicked(self._fetch_callback)

        # Asteroid Selector (using RadioButtons in a compact box)
        self.ax_controls.text(0.05, 0.78, "2. Select an Asteroid", transform=self.ax_controls.transAxes, fontsize=10, fontweight='bold')
        ax_radio_box = self.fig.add_axes([0.7, 0.45, 0.25, 0.32]) # Positioned for the list
        self.radio_buttons = widgets.RadioButtons(ax_radio_box, ["Fetch asteroids first..."])

        # Location Inputs
        self.ax_controls.text(0.05, 0.40, "3. Impact Location (or click map)", transform=self.ax_controls.transAxes, fontsize=10, fontweight='bold')
        ax_lat_box = self.fig.add_axes([0.7, 0.35, 0.25, 0.04])
        self.lat_textbox = widgets.TextBox(ax_lat_box, 'Latitude:', initial="40.7128")
        ax_lon_box = self.fig.add_axes([0.7, 0.30, 0.25, 0.04])
        self.lon_textbox = widgets.TextBox(ax_lon_box, 'Longitude:', initial="-74.0060")

        # Simulate Button
        ax_sim_btn = self.fig.add_axes([0.7, 0.24, 0.25, 0.04])
        self.simulate_button = widgets.Button(ax_sim_btn, 'Simulate Impact')
        self.simulate_button.on_clicked(self._simulate_callback)
        
        # Report Area
        self.report_text = self.fig.text(0.68, 0.02, "Impact report will be shown here.", fontsize=8, va='bottom')

    def _fetch_callback(self, event):
        api_key = self.api_textbox.text
        if not api_key:
            self.report_text.set_text("Error: Please enter an API key.")
            self.fig.canvas.draw_idle()
            return
        
        self.report_text.set_text("Fetching data...")
        self.fig.canvas.draw_idle()

        api = NasaNeoAPI(api_key)
        self.neos = api.fetch_recent_neos()
        
        if not self.neos:
            self.report_text.set_text("Failed to fetch data. Check API key.")
            self.fig.canvas.draw_idle()
            return

        # Update RadioButtons with new labels
        # Limiting to top 10 for a cleaner UI
        labels = [f"{neo['name']} ({neo['diameter_km']:.3f} km)" for neo in self.neos[:10]]
        if not labels:
            labels = ["No asteroids found."]
        
        self.radio_buttons.ax.clear()
        # Re-initialize RadioButtons in the same axes object
        self.radio_buttons._init_(self.radio_buttons.ax, labels)
        
        self.report_text.set_text(f"Fetched {len(self.neos)} NEOs. Select one.")
        self.fig.canvas.draw_idle()

    def _simulate_callback(self, event):
        if not self.neos or self.radio_buttons.value_selected is None:
            self.report_text.set_text("Error: Please fetch and select an asteroid.")
            return

        try:
            lat = float(self.lat_textbox.text)
            lon = float(self.lon_textbox.text)
        except ValueError:
            self.report_text.set_text("Error: Invalid latitude or longitude.")
            return

        # Find the selected NEO object from the list
        selected_label = self.radio_buttons.value_selected
        # The radio button labels are from the first 10 asteroids
        try:
            # We need to find the index from the labels list
            current_labels = [label.get_text() for label in self.radio_buttons.labels]
            selected_index = current_labels.index(selected_label)
            selected_neo = self.neos[selected_index]
        except (ValueError, IndexError):
            self.report_text.set_text("Error: Could not find selected asteroid.")
            return
        
        # Calculate effects
        calculator = ImpactCalculator(selected_neo)
        effects = calculator.get_effects()

        # Update report text
        report_str = (
            f"========= IMPACT REPORT =========\n"
            f"Asteroid: {selected_neo['name']}\n"
            f"Diameter: {selected_neo['diameter_km']:.3f} km\n"
            f"Impact Velocity: {selected_neo['velocity_kps']:.2f} km/s\n"
            f"---------------------------------\n"
            f"Impact Energy: {effects['energy_megatons_tnt']:,.2f} Megatons TNT\n"
            f"Crater (Fragmented): {effects['fragmented_crater_km']:.2f} km\n"
            f"Crater (Solid Body): {effects['solid_body_crater_km']:.2f} km\n"
            f"Fireball Radius: {effects['fireball_radius_km']:.2f} km\n"
            f"Earthquake Mag: {effects['earthquake_magnitude']:.2f} (Richter scale)"
        )
        self.report_text.set_text(report_str)

        # Update map
        self._draw_impact_on_map(lon, lat, effects)
        self.fig.canvas.draw_idle()

    def _draw_impact_on_map(self, lon, lat, effects):
        # Clear previous impact drawings and selection markers
        for artist in self.ax_map.patches + self.ax_map.lines:
            if hasattr(artist, 'get_label') and 'impact' in artist.get_label():
                artist.remove()
        
        if hasattr(self, 'click_marker') and self.click_marker in self.ax_map.lines:
            self.click_marker.remove()

        km_per_degree = 111.0
        
        # Function to create a circle patch
        def create_circle(radius_km, color, label):
            radius_deg = (radius_km / 2) / km_per_degree
            return plt.Circle((lon, lat), radius_deg, color=color, alpha=0.6, label=label)

        solid_crater = create_circle(effects['solid_body_crater_km'], 'saddlebrown', 'impact_solid_crater')
        frag_crater = create_circle(effects['fragmented_crater_km'], '#4d2600', 'impact_frag_crater')
        fireball = create_circle(effects['fireball_radius_km'] * 2, 'red', 'impact_fireball')
        
        self.ax_map.add_patch(fireball)
        self.ax_map.add_patch(solid_crater)
        self.ax_map.add_patch(frag_crater)
        self.ax_map.plot(lon, lat, 'rx', markersize=8, label='impact_point')
        
        # Auto-zoom to the impact area
        max_effect_radius_km = max(
            effects['solid_body_crater_km'] / 2,
            effects['fragmented_crater_km'] / 2,
            effects['fireball_radius_km']
        )
        view_radius_deg = (max_effect_radius_km * 1.5) / km_per_degree
        min_view_deg = 5.0
        if view_radius_deg < min_view_deg:
            view_radius_deg = min_view_deg
        
        self.ax_map.set_xlim(lon - view_radius_deg, lon + view_radius_deg)
        self.ax_map.set_ylim(lat - view_radius_deg, lat + view_radius_deg)


if __name__ == "_main_":
    app = AppGUI()