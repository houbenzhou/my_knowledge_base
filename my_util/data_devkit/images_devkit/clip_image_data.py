import rasterio
from rasterio.windows import Window


def clip_image_data(ds, block_xmin, block_ymin, tile_size_x, tile_size_y, transf_tile, outputdata):
    img = ds.read(
        window=Window(block_xmin, block_ymin, tile_size_x, tile_size_y))

    dst = rasterio.open(outputdata, 'w',
                        driver=ds.driver, width=tile_size_x, height=tile_size_y,
                        count=ds.count, bounds=ds.bounds, crs=ds.crs, transform=transf_tile,
                        dtype=ds.dtypes[0])
    dst.write(img)


if __name__ == '__main__':
    inputdata = '/home/data/hou/workspaces/iobjectspy/resources_ml/example/项目/中南勘测院/data/2020-04-17/mountain/识别.tif'
    # left =
    # top =
    # right =
    # bottom =
    outputdata = '/home/data/hou/workspaces/iobjectspy/resources_ml/example/项目/中南勘测院/data/2020-04-17/mountain/识别1.tif'
    with rasterio.open(inputdata) as ds:
        transf = ds.transform
        # ymin, xmin = rasterio.transform.rowcol(transf, left,
        #                                        top)
        # ymax, xmax = rasterio.transform.rowcol(transf, right,
        #                                        bottom)

        # xmin = 100
        # ymin = 100
        # tile_size_x = xmax - xmin
        # tile_size_y = ymax - ymin


        xmin = 1400
        ymin = 1000
        tile_size_x = 4120
        tile_size_y = 7000
        coord_min = rasterio.transform.xy(transf, int(ymin), int(xmin))
        coord_max = rasterio.transform.xy(transf, int(ymin + tile_size_y),
                                          int(xmin + tile_size_x))
        transf_tile = rasterio.transform.from_bounds(coord_min[0], coord_max[1],
                                                     coord_max[0], coord_min[1],
                                                     tile_size_x, tile_size_y)
        clip_image_data(ds, xmin, ymin, tile_size_x, tile_size_y, transf_tile, outputdata)
