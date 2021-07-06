library(tidyverse)

args = commandArgs(trailingOnly = TRUE)

plate <- args[1]
# plate <- '20210521-p01-KJG_641'
metadata_dir <- stringr::str_c('output', 'metadata', plate, sep = '/')
output_dir <- stringr::str_c('output', 'data', sep = '/')

# get the paths to all the metadata files
metadata_files <- dplyr::tibble(base = metadata_dir,
                                plate = plate,
                                category = list.files(path = metadata_dir,
                                                      pattern = ".*.csv$",
                                                      recursive = TRUE)) %>%
  dplyr::mutate(path = stringr::str_c(base, category, sep = '/'),
                assay_date = stringr::str_extract(plate, "20[0-9]{6}"),
                category = stringr::str_remove(category, '.csv') %>% stringr::str_remove(., 'metadata/')) %>%
  dplyr::select(path, assay_date, plate, category)


# function to read and tidy the metadata files
get_metadata <- function(...) {

  df <- tibble(...)

  data <- readr::read_csv(df$path, col_names = sprintf("%02d", seq(1:12))) %>%
    dplyr::mutate(row = LETTERS[1:8], .before = `01`) %>%
    tidyr::pivot_longer(cols = `01`:`12`, names_to = 'col', values_to = df$category) %>%
    dplyr::mutate(well = stringr::str_c(row, col), plate = df$plate)

}

collapse_rows <- function(x) {
  x <- na.omit(x)
  if (length(x) > 0) first(x) else NA
}

metadata <- metadata_files %>%
  purrr::pmap_dfr(get_metadata) %>%
  dplyr::select(plate, well, row, col, species, stages, strain, treatment, conc, other) %>%
  dplyr::group_by(plate, well, row, col) %>%
  dplyr::summarise(dplyr::across(species:other, collapse_rows))

output_df <- readr::read_csv(stringr::str_c(output_dir,
                                            stringr::str_remove(plate, '_[0-9]*$') %>% stringr::str_c(., '_straightened_worms.csv'),
                                            sep = '/')) %>%
  rename(well = Metadata_Well)

final_df <- dplyr::left_join(metadata, output_df) %>%
  readr::write_csv(path = stringr::str_c(output_dir, '/', plate, '_tidy.csv'))
