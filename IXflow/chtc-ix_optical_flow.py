import argparse
import string
import itertools
import cv2
import numpy as np
import csv
import glob
from pathlib import Path
import imageio
from PIL import Image
from skimage.filters import threshold_otsu
from skimage.transform import rescale
from skimage import filters
from matplotlib import cm
from scipy import ndimage
from datetime import datetime


def create_plate(plate):
    '''
    Create a list with all the well names for a given plate format.
    '''
    # create a list of the possible well names
    wells = []
    if plate == 96:
        rows = string.ascii_uppercase[:8]
        columns = list(range(1, 13))
        columns = [str(column).zfill(2) for column in columns]
        wells = list(itertools.product(rows, columns))
        wells = [''.join(well) for well in wells]
    # print(wells)

    return wells


def organize_arrays(input, output, work, plate, frames, rows, columns, reorganize):
    '''
    Create a list of lists where each internal list is all the paths to all the
    images for a given well, and the entire list is the plate.
    '''

    for well in plate:
        print("Getting the paths for well {}".format(well))

        # initialize a list that will contain paths to each frame
        well_paths = []

        # append the path pointing to each frame from each well
        for frame in range(1, frames + 1):
            dir = Path.home().joinpath(input)
            name = dir.name.split("_")[0]
            path = dir.joinpath(str(dir), "TimePoint_" +
                                str(frame), name + "_" + well + ".TIF")
            well_paths.append(path)

        # read the TIFFs at the end of each path into a np array; perform
        # various operations on the array
        try:
            # get the dimensions of the images and write the first frame for dx
            first_frame = Image.open(str(well_paths[0]))
            dir = Path.home().joinpath(work)
            plate_name = Path.home().joinpath(input)
            plate_name = plate_name.name.split("_")[0]
            dir.joinpath(well, 'img').mkdir(parents=True, exist_ok=True)
            outpath = dir.joinpath(well, 'img', plate_name + "_" + well + '_orig' + ".png")
            imageio.imwrite(str(outpath), first_frame)

            height, width = np.array(first_frame).shape

            # initialize an empty array with the correct shape of the final array
            well_array = np.zeros((frames, height, width))
            counter = 0

            # read images from well_paths
            print("Reading images for well {}".format(well))
            for frame in well_paths:
                image = Image.open(str(frame))
                matrix = np.array(image)
                well_array[counter] = matrix

                if reorganize:
                    counter_str = str(counter).zfill(2)
                    dir = Path.home().joinpath(work)
                    plate_name = Path.home().joinpath(input)
                    plate_name = plate_name.name.split("_")[0]
                    dir.joinpath(well, 'vid').mkdir(parents=True, exist_ok=True)
                    outpath = dir.joinpath(well, 'vid', plate_name + "_" + well + "_" + counter_str + ".tiff")
                    cv2.imwrite(str(outpath), matrix)

                counter += 1

            # run flow on the well
            total_sum, sum_img = dense_flow(
                well,
                well_array,
                input,
                output,
                work)

            # segment the worms
            normalization_factor, sobel, blur, bin = segment_worms(
                well,
                well_array,
                input,
                output,
                work)

            # wrap_up
            wrap_up(
                well,
                total_sum,
                normalization_factor,
                input,
                output)

            # add to the dict with the well as the key and the array as the value
            # vid_dict[well] = well_array

            # saving as 16 bit AVI not currently working
            # fourcc = cv2.VideoWriter_fourcc(*'FFV1')
            # outvid = dir.joinpath(well, plate_name + "_" + well + ".avi")
            # out = cv2.VideoWriter(str(outvid), fourcc, 4, (height,  width), False)
            # for frame in well_array:
            #     frame = frame.astype('uint8')
            #     out.write(frame)

        except FileNotFoundError:
            print("Well {} not found. Moving to next well.".format(well))
            counter += 1

    # return vid_dict


def dense_flow(well, well_array, input, output, work):
    '''
    Uses Farneback's algorithm to calculate optical flow for each well. To get
    a single motility values, the magnitude of the flow is summed across each
    frame, and then again for the entire array.
    '''

    start_time = datetime.now()
    print("Starting optical flow analysis on {}.".format(well))

    array = well_array

    length, width, height = array.shape

    # initialize emtpy array of video length minus one (the length of the dense flow output)
    all_mag = np.zeros((length - 1, height, width))
    count = 0
    frame1 = array[count]

    while(1):
        if count < length - 1:
            frame1 = array[count].astype('uint16')
            frame2 = array[count + 1].astype('uint16')

            flow = cv2.calcOpticalFlowFarneback(frame1, frame2, None, 0.5, 3,
                                                15, 3, 5, 1.2, 0)

            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

            frame1 = frame2

            # replace proper frame with the magnitude of the flow between prvs and next frames
            all_mag[count] = mag
            count += 1

        else:
            break

    # calculate total flow across the entire array
    sum_img = np.sum(all_mag, axis=0)
    total_sum = np.sum(sum_img)

    # write out the dx flow image
    dir = Path.home().joinpath(work)
    plate_name = Path.home().joinpath(input)
    plate_name = plate_name.name.split("_")[0]
    dir.joinpath(well, 'img').mkdir(parents=True, exist_ok=True)
    outpath = dir.joinpath(well, 'img', plate_name + "_" + well + '_flow' + ".png")

    # write to png
    print(str(outpath))
    imageio.imwrite(str(outpath), sum_img)

    print("Optical flow anlaysis completed. Analysis took {}".format(
        datetime.now() - start_time))

    print(total_sum)

    return total_sum, sum_img


def segment_worms(well, well_array, input, output, work):
    '''
    Segments worms to use for downstream normalization.
    '''

    start_time = datetime.now()
    print("Starting normalization calculation for {}.".format(well))

    array = well_array

    print("Segmenting 5th frame...")

    # sobel edge detection
    sobel = filters.sobel(array[4])

    # gaussian blur
    blur = ndimage.filters.gaussian_filter(sobel, 1.5)

    # set threshold, make binary
    threshold = threshold_otsu(blur)
    binary = blur > threshold

    dir = Path.home().joinpath(work)
    name = Path.home().joinpath(input)
    name = name.name.split("_")[0]
    outpath = dir.joinpath(well, 'img')

    sobel_png = dir.joinpath(outpath, name + "_" + well + '_edge' + ".png")
    imageio.imwrite(sobel_png, sobel)

    blur_png = dir.joinpath(outpath, name + "_" + well + '_blur' + ".png")
    imageio.imwrite(blur_png, blur)

    bin_png = dir.joinpath(outpath, name + "_" + well + '_binary' + ".png")
    imageio.imwrite(bin_png, binary.astype(int))

    print("Calculating normalization factor.")

    # the area is the sum of all the white pixels (1.0)
    area = np.sum(binary)
    print("Normalization factor calculation completed. Calculation took {} \
          seconds.".format(datetime.now() - start_time))

    return area, sobel, blur, binary


def wrap_up(well, motility, normalization_factor, input, output):
    '''
    Takes dictionaries of values and writes them to a CSV.
    '''

    dir = Path.home().joinpath(output)
    name = Path.home().joinpath(input)
    name = name.name.split("_")[0]
    outfile = dir.joinpath(name + '_data' + ".csv")

    with open(str(outfile), 'a') as of:
        writer = csv.writer(of, delimiter=',')
        writer.writerow([well, motility, normalization_factor])


def thumbnails(rows, cols, input, output, work):
    '''
    Takes a dict that contains a video, rescales into thumbnails, and pastes
    into the structure of the plate.
    '''

    # get the paths to all the dx images
    dir = Path.home().joinpath(work)

    orig_thumbs = {}
    orig_search = str(dir.joinpath('**/img/*_orig.png'))
    orig_paths = glob.glob(orig_search)

    flow_thumbs = {}
    flow_search = str(dir.joinpath('**/img/*_flow.png'))
    flow_paths = glob.glob(flow_search)

    binary_thumbs = {}
    binary_search = str(dir.joinpath('**/img/*_binary.png'))
    binary_paths = glob.glob(binary_search)

    # rescale images and store them in a dict with well/image
    def create_thumbs(paths, dict):
        for path in paths:
            well = path.split('/')[-3]
            image = imageio.imread(path)
            # rescale the imaging without anti-aliasing
            rescaled = rescale(image, 0.125, anti_aliasing=True)
            # normalize to 0-255
            rescaled_norm = cv2.normalize(src=rescaled, dst=None, alpha=0,
                                          beta=255, norm_type=cv2.NORM_MINMAX,
                                          dtype=cv2.CV_8U)
            dict[well] = rescaled_norm
        return dict

    # generate thumbnails for all dx images
    orig_thumbs = create_thumbs(orig_paths, orig_thumbs)
    flow_thumbs = create_thumbs(flow_paths, flow_thumbs)
    binary_thumbs = create_thumbs(binary_paths, binary_thumbs)

    # 0.125 of the 4X ImageXpress image is 256 x 256 pixels
    height = rows * 256
    width = cols * 256

    # stitch the thumbnails together with the proper plate dimensions and save
    def create_plate_thumbnail(thumbs, type):
        new_im = Image.new('L', (width, height))

        for well, thumb in thumbs.items():
            # row letters can be converted to integers with ord() and subtracting a constant
            row = int(ord(well[:1]) - 64)
            col = int(well[1:].strip())
            new_im.paste(Image.fromarray(thumb), ((col - 1) * 256, (row - 1) * 256))

        if type == 'flow':
            # apply a colormap if it's a flow image
            new_im = np.asarray(new_im) / 255
            new_im = Image.fromarray(np.uint8(cm.inferno(new_im)*255))

        dir = Path.home().joinpath(output)
        dir.joinpath('thumbs').mkdir(parents=True, exist_ok=True)
        plate_name = Path.home().joinpath(input)
        plate_name = plate_name.name.split("_")[0]
        outfile = dir.joinpath('thumbs', plate_name + '_' + type + ".png")

        new_im.save(outfile)

    create_plate_thumbnail(orig_thumbs, 'orig')
    create_plate_thumbnail(flow_thumbs, 'flow')
    create_plate_thumbnail(binary_thumbs, 'binary')


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='To be determined...')

    # required positional arguments
    parser.add_argument('input_directory',
                        help='A path to a directory containing subdirectories \
                        filled with TIFF files (i.e. 20201118-p01-MZ_172/     \
                        TimePoint_1, etc.)')

    parser.add_argument('output_directory',
                        help='A path to the output directory.')

    parser.add_argument('work_directory',
                        help='A path to the work directory.')

    parser.add_argument('rows', type=int,
                        help='The number of rows in the imaging plate.')

    parser.add_argument('columns', type=int,
                        help='The number of columns in the imaging plate.')

    parser.add_argument('time_points', type=int,
                        help='The number of frames recorded.')

    # optional flags
    parser.add_argument('--reorganize', dest='reorganize', action='store_true',
                        default=False,
                        help='Invoke if you want to save the TIFF files       \
                        organized by well instead of time point (default is   \
                        to not reorganize).')

    args = parser.parse_args()

    # create the plate shape
    plate_format = args.rows * args.columns

    # create a list of all the possible wells in the plate
    plate = create_plate(plate_format)

    # re-organize the input TIFFs so that each well has its own array
    vid_dict = organize_arrays(
        args.input_directory,
        args.output_directory,
        args.work_directory,
        plate,
        args.time_points,
        args.rows,
        args.columns,
        args.reorganize)

    thumbnails(
        args.rows,
        args.columns,
        args.input_directory,
        args.output_directory,
        args.work_directory)
