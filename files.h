#pragma once

#include <vector>
#include <string>

// parameters for comparison with ground truth
struct GTcheckParameters {
    GTcheckParameters () : gtCheck ( false ), noccCheck ( false ), scale ( 150.0f ), dispTolGT ( 0.5f ), divFactor ( 4.0f ) {}
    bool gtCheck;
    bool noccCheck;
    float scale; // scaling factor just for visualization of error
    float dispTolGT;
    float dispTolGT2;
    //division factor dependent on ground truth data (to get disparity value: imgValue/divFactor)
    float divFactor; //Middleburry small images: 4, big images third: 3, Kitti: 255
};

//pathes to input images (camera images, ground truth, ...)
struct InputFiles {
    InputFiles () : gt_filename ( "" ), gt_nocc_filename ( "" ), occ_filename ( "" ), gt_normal_filename ( "" ), calib_filename ( "" ), images_folder ( "" ), p_folder ( "" ), camera_folder ( "" ),krt_file(""), pmvs_folder("") {}
    std::vector<std::string> img_filenames; // input camera images (only filenames, path is set in images_folder), names can also be used for calibration data (e.g. for Strecha P, camera)
    std::string gt_filename; // ground truth image
    std::string gt_nocc_filename; // non-occluded ground truth image (as provided e.g. by Kitti)
    std::string occ_filename; // occlusion mask (binary map of all the points that are occluded) (as provided e.g. by Middleburry)
    std::string gt_normal_filename; // ground truth normal map (for Strecha)
    std::string calib_filename; // calibration file containing camera matrices (P) (as provided e.g. by Kitti)
    std::string images_folder; // path to camera input images
    std::string p_folder; // path to camera projection matrix P (Strecha)
    std::string camera_folder; // path to camera calibration matrix K (Strecha)
    std::string krt_file; // path to camera matrixes in middlebury format
    std::string bounding_folder; //path to bounding volume (Strecha)
    std::string seed_file; // path to bounding volume (Strecha)
    std::string pmvs_folder; // path to pmvs folder
};

//pathes to output files
struct OutputFiles {
    OutputFiles () : parentFolder ( "results" ), disparity_filename ( 0 ) {}
    const char* parentFolder;
    char* disparity_filename;
};
