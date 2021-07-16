#pragma once

#include <vector>
#include <limits>

namespace BM_Tracker {

    typedef int coordType;

    struct Joint {
        bool present = false;
        coordType x, y;
        Joint() {}
        Joint(coordType x, coordType y) : x(x), y(y), present(true) {}
    };

    struct PoseData {
        std::vector< Joint > joint; 
        //pose index
        int id = -1;
        //output
        int track = -1;

        //derived pose values
        coordType bboxMin[2], bboxMax[2];
        coordType centerX, centerY;
        coordType baseRadius;
        //currently assigned tracklet id (internal tracker variable)
        int trklId;

        void calcBbox() {
            for (int i = 0; i < 2; i++) {
                bboxMin[i] = std::numeric_limits<coordType>::max();
                bboxMax[i] = std::numeric_limits<coordType>::lowest();
            }
            for (auto& j : joint) {
                if (j.present) {
                    bboxMin[0] = std::min(bboxMin[0], j.x);
                    bboxMax[0] = std::max(bboxMax[0], j.x);
                    bboxMin[1] = std::min(bboxMin[1], j.y);
                    bboxMax[1] = std::max(bboxMax[1], j.y);
                }    
            }
        }

        PoseData(int jointCount = 0) : joint(jointCount) {}
    };

    struct Frame {
        std::vector< PoseData > poses;
    };


}
