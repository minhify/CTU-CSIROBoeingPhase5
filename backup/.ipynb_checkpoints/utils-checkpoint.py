import geopandas as gpd


def load_data_geo(path: str):
    gdf = gpd.read_file(path)
    return gdf