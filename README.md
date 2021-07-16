# High-Speed Multi-Person Tracking Method using Bipartite Matching
- [Description](#description)
- [Dependencies](#dependecies)
- [Usage](#usage)
- [Examples](#examples)

## Description

This repository containes the code for the paper "High-Speed Multi-Person Tracking Method using Bipartite Matching" presented at the SAI Computing Conference 2021.

Currently only an C++ implementation is available. We're planning to provide evaluation examples and Python bindings later.

## Dependencies

- C++ 11
- [OpenCV](https://opencv.org/) (for image input and color feature calculation, optional)
- [nlohmann-json](https://github.com/nlohmann/json/) (for configuration file loading, optional)

## Usage

The tracker is presented as an include-only C++ library. You can use it directly by adding the include folder to the include paths, and including `bm_tracker/tracker.h` in the source files.

### CMake

You can also use it as a CMake package, by embdedding this source tree in a subdirectory and call `add_subdirectory(...)`, `target_link_libraries(your_target_name PRIVATE bm_tracker)` in your `CMakeLists.txt` file. 
You can use CMake option `-DBM_TRACKER_FEATURES=OFF` to disable OpenCV dependency and remove `CColor` term support. To prevent it from downloading nlohmann::json automatically, you can use `-DBM_TRACKER_DOWNLOAD_JSON=OFF` flag, and to disable json loading `-DBM_TRACKER_JSON=OFF`.

## Examples

```c++
#include <bm_tracker/tracker.h>

using namespace BM_Tracker;

int main() {
    //loading tracker configuration from a json file
    TrackerConfig config {std::string("tests/sample_config.json") };
    Tracker tracker{config};

    config.jointCount = 18;     //number of joints in used pose model, e.g. 18 for COCO
    config.centerJoint = 1;     //joint used as a 'center' joint, e.g. neck
    while (hasFrame) {
        //...
        //detection code
        //acquire detectedPoses array with pose data
        //and processed video frame as cv::Mat frameImage
        //...

        Frame frame;
        for (auto &pose : detectedPoses) {
            PoseData poseData {config.jointCount};
            for (int i = 0; i < config.jointCount; i++) {
                if (pose.keypoint_present[i]) {
                    poseData.joint[i] = {pose.keypoint_x[i], pose.keypoint_y[i]};
                }
            }
            frame.poses.push_back(poseData);
        }
        tracker.process(frame, frameImage);
        //...
        //do something with obtained track ids at frame.poses[i].track
        //...
    }
}
```

