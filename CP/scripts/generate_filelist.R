library(tidyverse)

# setwd('~/Desktop/')

args = commandArgs(trailingOnly = TRUE)

plate <- args[1]
mask <- args[2]
# plate <- '20210521-p01-KJG_641'
# wd <-'Users/njwheeler/Desktop'
# mask <- 'well_mask.png'

image_dir <- stringr::str_c(getwd(), 'CellProfiler_Pipelines', 'projects', plate, 'raw_images', sep = '/')

input_files <- list.files(path = image_dir, pattern = '.*TIF$')

load_csv <- dplyr::tibble(
  Group_Number = 1,
  Group_Index = seq(1, length(input_files)),
  URL_RawImage = stringr::str_c('file:', getwd(), 'CellProfiler_Pipelines', 'projects', plate, 'raw_images', input_files, sep = '/'),
  URL_WellMask = stringr::str_c('file:', getwd(), 'CellProfiler_Pipelines', 'masks', mask, sep = '/'),
  PathName_RawImage = stringr::str_remove(URL_RawImage, pattern = "/[^/]*$") %>% str_remove(., 'file:'),
  PathName_WellMask = stringr::str_remove(URL_WellMask, mask) %>% str_remove(., 'file:') %>%  str_remove(., '/$'),
  FileName_RawImage = input_files,
  FileName_WellMask = mask,
  Series_RawImage = 0,
  Series_WellMask = 0,
  Frame_RawImage = 0,
  Frame_WellMask = 0,
  Channel_RawImage = -1,
  Channel_WellMask = -1,
  Metadata_Date = stringr::str_extract(plate, '202[0-9]{5}'),
  Metadata_FileLocation = URL_RawImage,
  Metadata_Frame = 0,
  Metadata_Plate = stringr::str_extract(plate, '-p[0-9]*-') %>% stringr::str_remove_all(., '-'),
  Metadata_Researcher = stringr::str_extract(plate, '-[A-Z]{2,3}') %>% stringr::str_remove_all(., '-'),
  Metadata_Series = 0,
  Metadata_Well = stringr::str_extract(FileName_RawImage, '[A-H][0,1]{1}[0-9]{1}')
)

readr::write_csv(load_csv, file = stringr::str_c('/', getwd(), '/CellProfiler_Pipelines/', 'metadata/', 'image_paths.csv', sep = ''))


