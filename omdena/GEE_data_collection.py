import ee
import folium
from folium import plugins
import pandas as pd
import geopandas as gpd
from typing import Dict, List, Union
import numpy as np
import matplotlib.pyplot as plt
import os

ee.Authenticate()

# Initialize Earth Engine
try:
    ee.Initialize()
except Exception as e:
    print("Error initializing Earth Engine. Make sure you are authenticated:", str(e))
    exit(1)

# Uttarakhand district names
def get_uttarakhand_districts() -> list:
    """Fetch list of Uttarakhand districts from Google Earth Engine."""
    districts = ee.FeatureCollection('FAO/GAUL/2015/level2')
    uttarakhand_districts = districts.filter(ee.Filter.eq('ADM1_NAME', 'Uttarakhand'))
    
    # Get the list of district names
    district_list = uttarakhand_districts.aggregate_array('ADM2_NAME').getInfo()
    return sorted(district_list)  # Sort alphabetically for better display

# Replace the static UTTARAKHAND_DISTRICTS with a function call
UTTARAKHAND_DISTRICTS = get_uttarakhand_districts()

# Data type options and their corresponding GEE datasets
DATA_TYPES = {
    'Land cover maps': 'ESA/WorldCover/v200',
    'Elevation data': ee.Image('USGS/SRTMGL1_003'),
    'Population density': 'CIESIN/GPWv411/GPW_Population_Density',
    'Water body information': ee.Image('JRC/GSW1_4/GlobalSurfaceWater')  # Fixed asset path
}

# Resolution ranges for different data types (in meters)
DATA_TYPE_RESOLUTIONS = {
    'Land cover maps': {'min': 10, 'max': 100},
    'Elevation data': {'min': 30, 'max': 100},
    'Population density': {'min': 100, 'max': 1000},
    'Water body information': {'min': 30, 'max': 100}
}

def load_shapefile(shapefile_path: str) -> ee.FeatureCollection:
    """
    Load and process a shapefile into an Earth Engine FeatureCollection.
    
    Args:
        shapefile_path (str): Path to the shapefile
        
    Returns:
        ee.FeatureCollection: The processed shapefile as an EE FeatureCollection
    """
    try:
        # Read the shapefile using geopandas
        gdf = gpd.read_file(shapefile_path)
        
        # Convert to GeoJSON format
        geojson = gdf.__geo_interface__
        
        # Create an EE FeatureCollection from the GeoJSON
        ee_mask = ee.FeatureCollection(geojson)
        
        return ee_mask
    except Exception as e:
        print(f"Error loading shapefile: {str(e)}")
        raise

def get_user_inputs() -> Dict[str, Union[str, int, str]]:
    """Get user inputs for data extraction."""
    # Display available districts
    print("\nAvailable districts in Uttarakhand:")
    for i, district in enumerate(UTTARAKHAND_DISTRICTS, 1):
        print(f"{i}. {district}")
    
    while True:
        try:
            district_idx = int(input("\nSelect district number: ")) - 1
            if 0 <= district_idx < len(UTTARAKHAND_DISTRICTS):
                selected_district = UTTARAKHAND_DISTRICTS[district_idx]
                break
            print("Invalid selection. Please try again.")
        except ValueError:
            print("Please enter a valid number.")

    # Display data type options
    print("\nAvailable data types:")
    data_types = list(DATA_TYPES.keys())
    for i, data_type in enumerate(data_types, 1):
        print(f"{i}. {data_type}")
    
    while True:
        try:
            type_idx = int(input("\nSelect data type number: ")) - 1
            if 0 <= type_idx < len(data_types):
                selected_type = data_types[type_idx]
                break
            print("Invalid selection. Please try again.")
        except ValueError:
            print("Please enter a valid number.")

    # Get resolution
    while True:
        try:
            resolution_range = DATA_TYPE_RESOLUTIONS[selected_type]
            resolution = float(input(f"\nEnter desired resolution in meters ({resolution_range['min']}-{resolution_range['max']}): "))
            if check_resolution(resolution, selected_type):
                break
            print(f"Resolution must be between {resolution_range['min']} and {resolution_range['max']} meters for {selected_type}. Please try again.")
        except ValueError:
            print("Please enter a valid number.")

    # Get shapefile path
    while True:
        shapefile_path = input("\nEnter the path to your shapefile (.shp): ")
        if shapefile_path.endswith('.shp') and os.path.exists(shapefile_path):
            break
        print("Invalid shapefile path. Please ensure the file exists and has .shp extension.")

    return {
        'district': selected_district,
        'data_type': selected_type,
        'resolution': resolution,
        'shapefile_path': shapefile_path
    }

def check_resolution(resolution: float, data_type: str) -> bool:
    """Check if the resolution is within valid bounds for the selected data type."""
    if data_type not in DATA_TYPE_RESOLUTIONS:
        return False
    
    resolution_range = DATA_TYPE_RESOLUTIONS[data_type]
    return resolution_range['min'] <= resolution <= resolution_range['max']

def get_district_boundary(district_name: str) -> ee.Feature:
    """Get the boundary of the selected district."""
    # Load India districts dataset
    districts = ee.FeatureCollection('FAO/GAUL/2015/level2')
    
    # Filter for the selected district in Uttarakhand
    district = districts.filter(ee.Filter.And(
        ee.Filter.eq('ADM1_NAME', 'Uttarakhand'),
        ee.Filter.eq('ADM2_NAME', district_name)
    )).first()
    
    # Check if district was found
    if district.getInfo() is None:
        raise ValueError(f"District '{district_name}' not found in Uttarakhand. Please check the district name.")
    
    return district

def extract_data(district: ee.Feature, data_type: str, resolution: float, mask_feature: ee.FeatureCollection = None) -> ee.Image:
    """
    Extract the selected data type for the district with optional masking.
    
    Args:
        district (ee.Feature): The district boundary
        data_type (str): Type of data to extract
        resolution (float): Desired resolution in meters
        mask_feature (ee.FeatureCollection, optional): Shapefile mask to apply
        
    Returns:
        ee.Image: The processed and masked image
    """
    # Get the dataset
    dataset = DATA_TYPES[data_type]
    
    if isinstance(dataset, str):
        image = ee.ImageCollection(dataset).mosaic()
    else:
        image = dataset
    
    # Clip to district boundary
    clipped_image = image.clip(district.geometry())
    
    # Apply additional mask if provided
    if mask_feature is not None:
        clipped_image = clipped_image.clip(mask_feature)
    
    return clipped_image.setDefaultProjection(crs='EPSG:4326', scale=resolution)

def visualize_data(image: ee.Image, district: ee.Feature, data_type: str):
    """Create visualization of the extracted data."""
    # Create a folium map centered on the district
    district_coords = district.geometry().centroid().coordinates().getInfo()
    map_center = [district_coords[1], district_coords[0]]
    
    m = folium.Map(location=map_center, zoom_start=9)
    
    # Add the data layer with appropriate visualization parameters
    vis_params = {
        'Land cover maps': {
            'bands': ['Map'],
            'min': 10,
            'max': 100,
            'palette': [
                '#006400',  # Tree cover
                '#ffbb22',  # Shrubland
                '#ffff4c',  # Grassland
                '#f096ff',  # Cropland
                '#fa0000',  # Built-up
                '#b4b4b4',  # Bare/sparse vegetation
                '#f0f0f0',  # Snow and ice
                '#0064c8',  # Water bodies
                '#0096a0',  # Herbaceous wetland
                '#00cf75',  # Mangroves
                '#fae6a0'   # Moss and lichen
            ]
        },
        'Elevation data': {'min': 0, 'max': 3000, 'palette': ['006633', 'E5FFCC', '662A00', 'D8D8D8', 'F5F5F5']},
        'Population density': {'min': 0, 'max': 1000, 'palette': ['yellow', 'orange', 'red']},
        'Water body information': {
            'min': 0,
            'max': 100,
            'palette': ['white', '#d3d3d3', '#a9a9a9', '#87ceeb', '#4682b4', '#0000ff']
        }
    }
    
    map_id_dict = image.getMapId(vis_params[data_type])
    folium.TileLayer(
        tiles=map_id_dict['tile_fetcher'].url_format,
        attr='Google Earth Engine',
        overlay=True,
        name=data_type
    ).add_to(m)
    
    # Add district boundary
    district_geom = district.geometry().getInfo()
    district_name = district.get('ADM2_NAME').getInfo()
    
    style_function = lambda x: {
        'fillColor': 'transparent',
        'color': '#000000',
        'weight': 2
    }
    
    folium.GeoJson(
        district_geom,
        name='District Boundary',
        style_function=style_function
    ).add_to(m)
    
    # Add district label
    folium.Popup(
        f'<b>{district_name}</b>',
        permanent=True
    ).add_to(folium.CircleMarker(
        location=map_center,
        radius=0,
        weight=0,
        fill=False
    ).add_to(m))
    
    # # Add scale bar
    # Scale().add_to(m)
    
    # Add minimap
    minimap = folium.plugins.MiniMap(toggle_display=True)
    m.add_child(minimap)
    
    # Add legend based on data type
    if data_type == 'Land cover maps':
        legend_items = [
            ('Tree cover', '#006400'),
            ('Shrubland', '#ffbb22'),
            ('Grassland', '#ffff4c'),
            ('Cropland', '#f096ff'),
            ('Built-up', '#fa0000'),
            ('Bare/sparse vegetation', '#b4b4b4'),
            ('Snow and ice', '#f0f0f0'),
            ('Water bodies', '#0064c8'),
            ('Herbaceous wetland', '#0096a0'),
            ('Mangroves', '#00cf75'),
            ('Moss and lichen', '#fae6a0')
        ]
    elif data_type == 'Elevation data':
        legend_items = [
            ('High elevation', '#F5F5F5'),
            ('Medium-high', '#D8D8D8'),
            ('Medium', '#662A00'),
            ('Medium-low', '#E5FFCC'),
            ('Low elevation', '#006633')
        ]
    elif data_type == 'Population density':
        legend_items = [
            ('High density', 'red'),
            ('Medium density', 'orange'),
            ('Low density', 'yellow')
        ]
    elif data_type == 'Water body information':
        legend_items = [
            ('No water', 'white'),
            ('Permanent water', '#d3d3d3'),
            ('Seasonal water', '#a9a9a9'),
            ('Rare water', '#87ceeb'),
            ('Frequent water', '#4682b4'),
            ('Permanent water body', '#0000ff')
        ]
    
    # Create and add the legend
    legend_html = '''
        <div style="position: fixed; 
                    bottom: 50px; right: 50px; width: 150px; height: auto;
                    border:2px solid grey; z-index:9999; font-size:14px;
                    background-color:white;
                    padding: 10px;
                    opacity: 0.8;">
        <div style="text-align: center; margin-bottom: 5px;"><b>Legend</b></div>
    '''
    
    for label, color in legend_items:
        legend_html += f'''
            <div style="display: flex; align-items: center; margin-bottom: 5px;">
                <div style="background-color: {color};
                            width: 20px; height: 20px;
                            margin-right: 5px;"></div>
                <div>{label}</div>
            </div>
        '''
    
    legend_html += '</div>'
    m.get_root().html.add_child(folium.Element(legend_html))

    # Add layer control
    folium.LayerControl().add_to(m)
    
    # Save the map
    map_file = f'uttarakhand_{data_type.lower().replace(" ", "_")}.html'
    m.save(map_file)
    print(f"\nMap saved as: {map_file}")
    
def summarize_data(image: ee.Image, district: ee.Feature, data_type: str):
    """Generate summary statistics for the extracted data."""
    stats = image.reduceRegion(
        reducer=ee.Reducer.mean().combine(
            ee.Reducer.minMax(), '', True
        ),
        geometry=district.geometry(),
        scale=30,
        maxPixels=1e9
    ).getInfo()
    
    print(f"\nSummary Statistics for {data_type}:")
    for key, value in stats.items():
        print(f"{key}: {value}")

def export_to_geotiff(image: ee.Image, district: ee.Feature, data_type: str, output_dir: str = './output'):
    """
    Export the image data as a GeoTIFF file.
    
    Args:
        image (ee.Image): The image to export
        district (ee.Feature): The district boundary
        data_type (str): Type of data being exported
        output_dir (str): Directory to save the GeoTIFF file
    """
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate filename based on data type and district name
        district_name = district.get('ADM2_NAME').getInfo()
        filename = f"{district_name}_{data_type.replace(' ', '_').lower()}.tif"
        output_path = os.path.join(output_dir, filename)
        
        # Get the geometry bounds for export
        geometry = district.geometry()
        if isinstance(geometry, ee.Geometry):
            geometry = geometry.bounds()
        
        # Set up the export task
        task = ee.batch.Export.image.toDrive(
            image=image,
            description=f'Export_{district_name}_{data_type}',
            folder='Earth_Engine_Exports',
            fileNamePrefix=filename.replace('.tif', ''),
            scale=30,  # You can adjust this based on your needs
            region=geometry,
            fileFormat='GeoTIFF',
            maxPixels=1e13
        )
        
        # Start the export task
        task.start()
        print(f"\nExport task started. The GeoTIFF will be saved to your Google Drive in the 'Earth_Engine_Exports' folder.")
        print(f"Filename: {filename}")
        
    except Exception as e:
        print(f"Error during export: {str(e)}")

def main():
    """Main function to run the data extraction and visualization pipeline."""
    try:
        # Get user inputs
        inputs = get_user_inputs()
        
        # Get district boundary
        district = get_district_boundary(inputs['district'])
        
        # Load and process shapefile if provided
        mask_feature = load_shapefile(inputs['shapefile_path'])
        
        # Extract data with mask
        image = extract_data(
            district=district,
            data_type=inputs['data_type'],
            resolution=inputs['resolution'],
            mask_feature=mask_feature
        )
        
        # Visualize and summarize data
        visualize_data(image, district, inputs['data_type'])
        summarize_data(image, district, inputs['data_type'])
        
        # Export data as GeoTIFF
        export_to_geotiff(image, district, inputs['data_type'])
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()