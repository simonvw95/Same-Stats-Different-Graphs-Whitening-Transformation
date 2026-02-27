from PIL import Image
from pdf2image import convert_from_path


def create_gifs(images, name):

    # create the frames
    frames = []
    imgs = images
    for i in imgs:
        # new_frame = Image.open(i)
        new_frame = convert_from_path(i)[0]
        frames.append(new_frame)

    rev = frames[::-1]
    rev.pop(0)
    rev.pop(-1)

    frames += rev

    frames[0].save("dance_gifs/{}.gif".format(name), format='GIF',
                append_images=frames[1:],
                save_all=True,
                duration=1000//10, loop=0)

    print('done')


cnts = list(range(1, 25))

# image_names = ['datadance_doubledozen/results/datadance_250_{}_datadance_250_{}/dddd_datadance_250_{}.pdf'.format(cnt, cnt, cnt) for cnt in cnts]
# create_gifs(image_names, 'dddd_implem-datadance250')

# image_names = ['datasaurus_dozen/results/random_cloud_n250_xm54-265_ym47-835_xsd16-765_ysd26-935_pc-0-065_datadance_250_{}/datadance_250_{}.pdf'.format(cnt, cnt, cnt) for cnt in cnts]
# create_gifs(image_names, 'dd_implem-datadance250')

# image_names = ['sgd/results/random_cloud_n250_xm54-265_ym47-835_xsd16-765_ysd26-935_pc-0-065_datadance_250_{}/random_cloud_n250_xm54-265_ym47-835_xsd16-765_ysd26-935_pc-0-065.pdf'.format(cnt) for cnt in cnts]
# create_gifs(image_names, 'sgd_implem-datadance250')

image_names = ['project/results/datadance_250_{}_xm54-265_ym47-835_xsd16-765_ysd26-935_pc-0-065/datadance_250_{}_xsd16-765_pc-0-065.pdf'.format(cnt, cnt) for cnt in cnts]
create_gifs(image_names, 'project_implem-datadance250')
