import os
import json
import glob
import numpy as np
import processing
from qgis.core import (
    QgsProject, QgsRasterLayer, QgsColorRampShader, 
    QgsRasterShader, QgsHillshadeRenderer, 
    QgsSingleBandPseudoColorRenderer, QgsVectorLayer,
    QgsSymbol, QgsPalLayerSettings, 
    QgsVectorLayerSimpleLabeling, QgsTextFormat, QgsTextBufferSettings, 
    QgsWkbTypes, QgsSingleSymbolRenderer, QgsUnitTypes, Qgis
)
from qgis.utils import iface
from qgis.PyQt.QtGui import QColor, QPainter, QFont

# =====================================================================
# CONFIGURATION: PATHS & CONTOUR STYLING
# =====================================================================
INPUT_RASTER_SUBDIR = "Surfaces"
MASTER_OUTPUT_SUBDIR = "Sitemap"
SITE_MAP_GROUP_NAME = "Site Map"  # Base name; auto-increments if it already exists

# Styling for all standard contours (Major, Intermediate, Minor)
UNIFORM_CONTOUR_COLOR = "#000000"
UNIFORM_CONTOUR_WIDTH = 0.40

# Styling specifically for the smallest/finest contour layer (first in config)
SMALLEST_CONTOUR_COLOR = "#8f7563"
SMALLEST_CONTOUR_WIDTH = 0.1
# =====================================================================

def run_automated_site_map_builder(tif_path, export_json=True):
    project = QgsProject.instance()
    dem_layer_name = os.path.splitext(os.path.basename(tif_path))[0]
    
    # Resolve Master Output Folder dynamically from Project Home
    project_home = project.homePath()
    if not project_home:
        raise ValueError("QGIS Project home path is not set. Please save your project file first.")
        
    master_output_folder = os.path.join(project_home, MASTER_OUTPUT_SUBDIR)
    site_output_folder = os.path.join(master_output_folder, f"{dem_layer_name} Site Map")
    os.makedirs(site_output_folder, exist_ok=True)
    print(f"Saving all outputs to site folder: {site_output_folder}")

    # --- 1. Find 'Site Location' Polygon Layer in the Project ---
    site_boundary_layer = None
    for layer in project.mapLayers().values():
        if layer.name() == "Site Location" and layer.geometryType() == QgsWkbTypes.PolygonGeometry:
            site_boundary_layer = layer
            break

    if not site_boundary_layer:
        raise ValueError("Error: Polygon layer named 'Site Location' not found in the QGIS project.")

    print(f"Found 'Site Location' layer: {site_boundary_layer.id()}. Calculating local stats & area.")

    # --- 2. Calculate Site Area in Hectares ---
    total_area_sqm = sum([feat.geometry().area() for feat in site_boundary_layer.getFeatures()])
    area_ha = total_area_sqm / 10000.0

    # --- 3. Initial check to get CRS ---
    temp_load = QgsRasterLayer(tif_path, dem_layer_name)
    if not temp_load.isValid():
        raise ValueError(f"Failed to load raster layer from path: {tif_path}")
    crs = temp_load.crs()

    # --- 4. Temporarily mask raster to 'Site Location' to calculate Z-range ---
    temp_masked_path = os.path.join(site_output_folder, "temp_site_stats_mask.tif")
    processing.run("gdal:cliprasterbymasklayer", {
        'INPUT': tif_path,
        'MASK': site_boundary_layer,
        'SOURCE_CRS': crs,
        'TARGET_CRS': crs,
        'CROP_TO_CUTLINE': True,
        'KEEP_RESOLUTION': True,
        'NODATA': -9999,
        'OUTPUT': temp_masked_path
    })

    temp_dem_layer = QgsRasterLayer(temp_masked_path, dem_layer_name)
    if not temp_dem_layer.isValid():
        raise ValueError("Failed to process temporary site-masked raster for statistics.")

    dem_provider = temp_dem_layer.dataProvider()
    cell_size_x = temp_dem_layer.rasterUnitsPerPixelX()
    cell_size_y = temp_dem_layer.rasterUnitsPerPixelY()
    cell_size = (cell_size_x + cell_size_y) / 2.0

    extent = temp_dem_layer.extent()
    width = temp_dem_layer.width()
    height = temp_dem_layer.height()
    
    block = dem_provider.block(1, extent, width, height)
    nodata_val = dem_provider.sourceNoDataValue(1)

    qgis_dt = dem_provider.dataType(1)
    dt_map = {
        Qgis.Byte: np.uint8,
        Qgis.UInt16: np.uint16,
        Qgis.Int16: np.int16,
        Qgis.UInt32: np.uint32,
        Qgis.Int32: np.int32,
        Qgis.Float32: np.float32,
        Qgis.Float64: np.float64
    }
    np_dtype = dt_map.get(qgis_dt, np.float32)

    ptr = block.data()
    bw = block.width()
    bh = block.height()
    
    elev = np.frombuffer(ptr, dtype=np_dtype).reshape((bh, bw)).astype(np.float32)
    
    valid_mask = np.isfinite(elev)
    if nodata_val is not None:
        valid_mask &= ~np.isclose(elev, nodata_val)
        
    valid_elev = elev[valid_mask]
    if valid_elev.size == 0:
        raise ValueError("No valid elevation data found in DEM within the 'Site Location' boundary.")

    px, py = np.gradient(elev, cell_size_x, cell_size_y)
    slope_deg = np.degrees(np.arctan(np.sqrt(px**2 + py**2)))
    valid_slope = slope_deg[valid_mask]

    z_min = float(np.min(valid_elev))
    z_max = float(np.max(valid_elev))
    z_range = z_max - z_min
    
    z_p02 = float(np.percentile(valid_elev, 2))
    z_p50 = float(np.percentile(valid_elev, 50))
    z_p98 = float(np.percentile(valid_elev, 98))
    
    mean_slope = float(np.mean(valid_slope))
    relief_type = "Steep/Mountainous" if mean_slope > 15.0 else "Rolling/Flat"

    # --- Calculate Dynamic Azimuth from Terrain Aspect ---
    mean_px = np.mean(px[valid_mask])
    mean_py = np.mean(py[valid_mask])
    mean_aspect = np.degrees(np.arctan2(-mean_px, mean_py)) % 360
    dynamic_azimuth = (mean_aspect + 135.0) % 360.0
    if relief_type == "Rolling/Flat" or np.isnan(dynamic_azimuth):
        dynamic_azimuth = 315.0 
    print(f"Calculated Dynamic Hillshade Azimuth: {dynamic_azimuth:.1f}°")

    del temp_dem_layer
    if os.path.exists(temp_masked_path):
        try:
            os.remove(temp_masked_path)
        except OSError:
            pass

    # --- Strict Contour Configuration Rules ---
    allow_large_contours = (z_range >= 200.0 or area_ha > 5.0)
    print(f"Site Relief Range: {z_range:.2f}m | Area: {area_ha:.2f} ha | Allow >10m Contours: {allow_large_contours}")

    if z_range < 5.0:
        contours_config = [
            {"name": "Minor", "interval": 0.1, "label": False},
            {"name": "Major", "interval": 0.5, "label": True}
        ]
    elif z_range < 10.0:
        contours_config = [
            {"name": "Minor", "interval": 0.2, "label": False},
            {"name": "Major", "interval": 1.0, "label": True}
        ]
    elif z_range < 25.0:
        contours_config = [
            {"name": "Minor", "interval": 0.2, "label": False},
            {"name": "Intermediate", "interval": 1.0, "label": True},
        ]
    elif z_range < 50.0:
        contours_config = [
            {"name": "Minor", "interval": 0.2, "label": False},
            {"name": "Intermediate", "interval": 1.0, "label": False},
            {"name": "Major", "interval": 5.0, "label": True}
        ]
    elif z_range < 200.0 and not allow_large_contours:
        contours_config = [
            {"name": "Ultra-Fine", "interval": 0.2, "label": False},
            {"name": "Minor", "interval": 1.0, "label": False},
            {"name": "Intermediate", "interval": 2.0, "label": True},
            {"name": "Major", "interval": 10.0, "label": True}
        ]
    else:
        if z_range <= 300.0:
            contours_config = [
                {"name": "Ultra-Fine", "interval": 1.0, "label": False},
                {"name": "Minor", "interval": 5.0, "label": False},
                {"name": "Intermediate", "interval": 20.0, "label": True},
                {"name": "Major", "interval": 100.0, "label": True}
            ]
        else:
            contours_config = [
                {"name": "Ultra-Fine", "interval": 2.0, "label": False},
                {"name": "Minor", "interval": 10.0, "label": False},
                {"name": "Intermediate", "interval": 50.0, "label": True},
                {"name": "Major", "interval": 200.0, "label": True}
            ]

    # Export Profile JSON
    if export_json:
        profile = {
            "dem": {
                "layer_name": dem_layer_name,
                "crs": crs.authid(),
                "units": crs.mapUnits().name,
                "resolution_gsd": cell_size,
                "elevation_stats": {
                    "min": z_min, "max": z_max, "z_range": z_range,
                    "area_ha": area_ha, "p02_clip": z_p02, 
                    "p50": z_p50, "p98_clip": z_p98, "mean": float(np.mean(valid_elev))
                },
                "terrain_metrics": {
                    "mean_slope_deg": mean_slope, 
                    "relief_type": relief_type,
                    "hillshade_azimuth": dynamic_azimuth,
                    "active_layers": [cfg["name"] for cfg in contours_config]
                }
            }
        }
        json_out = os.path.join(site_output_folder, "Site Map Profile.json")
        with open(json_out, "w") as f:
            json.dump(profile, f, indent=2)

    def apply_pseudocolor(target_layer, color_items):
        shader = QgsColorRampShader()
        shader.setColorRampType(QgsColorRampShader.Interpolated)
        shader.setColorRampItemList(color_items)
        raster_shader = QgsRasterShader()
        raster_shader.setRasterShaderFunction(shader)
        renderer = QgsSingleBandPseudoColorRenderer(
            target_layer.dataProvider(), 1, raster_shader
        )
        target_layer.setRenderer(renderer)

    # Base Rasters Setup (Unclipped full raster)
    slope_out_path = os.path.join(site_output_folder, "Site Map Slope Mask.tif")
    slope_res = processing.run("native:slope", {
        'INPUT': tif_path, 'Z_FACTOR': 1.5, 'OUTPUT': slope_out_path
    })
    slope_layer = QgsRasterLayer(slope_res['OUTPUT'], "Site Map Slope Mask")
    apply_pseudocolor(slope_layer, [
        QgsColorRampShader.ColorRampItem(0.0, QColor("#ffffff"), "0° (Flat)"),
        QgsColorRampShader.ColorRampItem(45.0, QColor("#222222"), "≥45° (Steep Drop-off)")
    ])
    slope_layer.setBlendMode(QPainter.CompositionMode_Overlay)
    slope_layer.setOpacity(0.25)

    multi_hs = QgsRasterLayer(tif_path, "Site Map Dynamic Hillshade")
    altitude = 35.0 if relief_type == "Steep/Mountainous" else 45.0
    multi_renderer = QgsHillshadeRenderer(multi_hs.dataProvider(), 1, dynamic_azimuth, altitude)
    multi_renderer.setMultiDirectional(False)
    multi_renderer.setZFactor(2.0)
    multi_hs.setRenderer(multi_renderer)
    multi_hs.setBlendMode(QPainter.CompositionMode_Multiply)
    multi_hs.setOpacity(0.20)

    # --- Setup Layer Tree Groups (Auto-incrementing unique name) ---
    root = project.layerTreeRoot()
    target_group_name = SITE_MAP_GROUP_NAME
    counter = 1
    while root.findGroup(target_group_name) is not None:
        target_group_name = f"{SITE_MAP_GROUP_NAME} {counter}"
        counter += 1
        
    site_group = root.addGroup(target_group_name)

    contour_group_name = "Contours"
    contour_group = site_group.findGroup(contour_group_name)
    if not contour_group:
        contour_group = site_group.addGroup(contour_group_name)

    # Add Derived Layers into Site Group
    project.addMapLayer(slope_layer, False)
    site_group.addLayer(slope_layer)

    project.addMapLayer(multi_hs, False)
    site_group.addLayer(multi_hs)

    # --- Generate, Style, Label, and Add Unclipped Contour Layers ---
    for index, cfg in enumerate(contours_config):
        layer_name = f"{cfg['name']} Contours ({cfg['interval']}m)"
        out_path = os.path.join(site_output_folder, f"Site Map {cfg['name']} Contours.gpkg")

        processing.run("gdal:contour", {
            'INPUT': tif_path,
            'BAND': 1,
            'INTERVAL': float(cfg['interval']),
            'FIELD_NAME': 'ELEV',
            'CREATE_3D': False,
            'IGNORE_NODATA': True,
            'NODATA': None,
            'OFFSET': 0.0,
            'EXTRA': '',
            'OUTPUT': out_path
        })

        vector_layer = QgsVectorLayer(out_path, layer_name, "ogr")

        if QgsWkbTypes.hasZ(vector_layer.wkbType()):
            vector_layer.startEditing()
            for feat in vector_layer.getFeatures():
                geom = feat.geometry()
                geom.get().dropZValue()
                vector_layer.changeGeometry(feat.id(), geom)
            vector_layer.commitChanges()

        symbol = QgsSymbol.defaultSymbol(vector_layer.geometryType())
        
        # Apply style: smallest contour (index 0) gets distinct style; all others share uniform style
        if index == 0:
            symbol.setColor(QColor(SMALLEST_CONTOUR_COLOR))
            symbol.setWidth(SMALLEST_CONTOUR_WIDTH)
        else:
            symbol.setColor(QColor(UNIFORM_CONTOUR_COLOR))
            symbol.setWidth(UNIFORM_CONTOUR_WIDTH)
            
        vector_layer.setRenderer(QgsSingleSymbolRenderer(symbol))

        if cfg['label']:
            label_settings = QgsPalLayerSettings()
            label_settings.fieldName = "ELEV"
            label_settings.isExpression = False
            label_settings.placement = Qgis.LabelPlacement.Line
            
            # Forces labels directly onto the vector path line
            label_settings.linePlacementFlags = QgsPalLayerSettings.OnLine
            label_settings.upsideDownLabels = Qgis.UpsideDownLabelHandling.AlwaysAllowUpsideDown

            text_format = QgsTextFormat()
            font_size = 3.3 if cfg['name'] == "Major" else 2.5
            text_format.setFont(QFont("Times New Roman", int(font_size), QFont.Bold))
            text_format.setSizeUnit(QgsUnitTypes.RenderMillimeters)
            text_format.setSize(font_size)
            text_format.setColor(QColor("#000000"))

            buffer_settings = QgsTextBufferSettings()
            buffer_settings.setEnabled(True)
            buffer_settings.setSizeUnit(QgsUnitTypes.RenderMillimeters)
            buffer_settings.setSize(0.8)
            buffer_settings.setColor(QColor("#ffffff"))
            text_format.setBuffer(buffer_settings)

            label_settings.setFormat(text_format)
            vector_layer.setLabeling(QgsVectorLayerSimpleLabeling(label_settings))
            vector_layer.setLabelsEnabled(True)

        project.addMapLayer(vector_layer, False)
        contour_group.addLayer(vector_layer)
        vector_layer.triggerRepaint()

    if iface:
        iface.mapCanvas().refresh()

    print(f"Success! Generated unclipped layers inside group '{target_group_name}'.")

# --- BATCH FOLDER ITERATION EXECUTION USING PROJECT HOME ---
project = QgsProject.instance()
project_home = project.homePath()

if not project_home:
    print("WARNING: Project home path is empty. Please save your project file first so batch processing can locate input folders.")
else:
    input_raster_folder = os.path.join(project_home, INPUT_RASTER_SUBDIR)
    
    for tif_path in glob.glob(os.path.join(input_raster_folder, "*.tif")):
        layer_name = os.path.splitext(os.path.basename(tif_path))[0]
        print(f"\n--- Processing Site Map Pipeline for: {layer_name} ---")
        run_automated_site_map_builder(tif_path, export_json=True)