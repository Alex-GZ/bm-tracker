#pragma once

#include "frame.h"

#include <vector>
#include <map>
#include <fstream>
#include <algorithm>

#include <assert.h>
#include <math.h>

#ifndef BM_TRACKER_DISABLE_FEATURES
#include "features.h"
#endif

#ifndef BM_TRACKER_DISABLE_JSON
#include <nlohmann/json.hpp>
#endif

namespace BM_Tracker {

#ifdef BM_TRACKER_DISABLE_FEATURES
    typedef void* matType; 
#endif

    struct TrackerConfig {
        int jointCount = 25;
        int centerJoint = 1;

        int trkSize = 30;
        int maxLimbDelay = 4;

        float cNew = 1.0f;
        float cPose = 0.4f;
        float cPoseMaxR = 0.5f;
        float cCenter = 0.6f / 2.5f;

        float cColor = 0.0f;

        float clrAlpha = 0.8f;
        float clrBeta = 1.5f;
        bool clrUseSqr = false;

        float cPckh = 0.0f;
        float cPckhMaxR = 0.1f;

        float cIOU = 0.0f;

        float wRationalBeta = 0.1f;

        bool alwaysDeriveCenter = false;

        TrackerConfig() {}
#ifndef BM_TRACKER_DISABLE_JSON

        TrackerConfig ( const nlohmann::json &trkc ) {
            alwaysDeriveCenter |= trkc.value("AlwaysDeriveCenter", false);
            
            cNew = trkc.value("CNew", 1.0f);
            cPose = trkc.value("CPose", 0.4f);
            cPoseMaxR = trkc.value("CPoseMaxR", 0.5f);
            cIOU = trkc.value("CIou", 0.0f);
            cPckh = trkc.value("CPckh", 0.0f);
            cPckhMaxR = trkc.value("CPckhMaxR", 0.1f);
            cCenter = trkc.value("CCenter", 0.6f / 2.5f);
            cColor = trkc.value("CColor", 0.0f);

            clrAlpha = trkc.value("ColorAlpha", 0.8f);
            clrBeta = trkc.value("ColorBeta", 1.5f);
            clrUseSqr = trkc.value("ColorSqr", false);

            wRationalBeta = trkc.value("WRationalBeta", 0.1f);
            
            trkSize = trkc.value("TrkSize", 30);
            maxLimbDelay = trkc.value("CPoseMaxDelay", 4);
        }

        TrackerConfig ( const std::string& path) {
            nlohmann::json configJson;
            std::ifstream configStream(path);
            configStream >> configJson;
        }
#endif
    };


    class Tracker {
    private:
        struct Track {
    #ifndef BM_TRACKER_DISABLE_FEATURES
            FeatureTrack fTrack;
    #endif
            TrackerConfig* config;

            struct JointState {
                coordType c[2];
                bool present = false;
            };

            struct State {
                std::vector< JointState > js;
                coordType center[2];
                bool hasCenter = false, present = false;
                float meterSize;
            };
            std::vector<State> states;

            float estA[2], estB[2];
            bool hasLinearEstimate  = false;
            int estT = 0;

            int historyId = -1;

            int id;
            int lastFrame;
            
            int trkSize = 0;
            int curI = 0;                  //index to write to at current frame

            int trkUid = -1;
            bool expired = false;
            int missingCount = 0;
            float estMeterSize;

            int curPose;

            inline float offsetToWeight(int off) {
                return 1.0f / (1.0f + (float) off * config->wRationalBeta);
            }

            void update_estimate(int c_frame) {
                int n = 0;
                estMeterSize = 0.0f;
                float total_w = 0.0f;
                for (int j = 0; j < std::min(trkSize, (int)states.size()); j++) {
                    int t_j = curI - 1 - j;
                    while (t_j < 0) t_j += (int)states.size();
                    if (states[t_j].hasCenter) {
                        n++;
                        if (n == 1) {
                            estB[0] = states[t_j].center[0];
                            estB[1] = states[t_j].center[1];
                        }
                        float w = offsetToWeight(j);
                        estMeterSize += states[t_j].meterSize * w;
                        total_w += w;
                    }
                }

                if (n > 1) {
                    estMeterSize /= total_w;
                    estT = c_frame;

                    float summ_t=0.0f, summ_t2=0.0f;
                    for (int j = 0; j < std::min(trkSize, (int)states.size()); j++) {
                        int t_j = curI - 1 - j;
                        while (t_j < 0) t_j += (int)states.size();
                        if (states[t_j].hasCenter) {
                            float w =  offsetToWeight(j);
                            summ_t += (float)(-j) * w;
                            summ_t2 += (float)(-j)*(-j) * w;
                        }
                    }
                    
                    for (int c = 0; c < 2; c++) {
                        float summ_tv=0.0f, summ_v2=0.0f, summ_v = 0.0f;

                        for (int j = 0; j < std::min(trkSize, (int)states.size()); j++) {
                            int t_j = curI - 1 - j;
                            while (t_j < 0) t_j += (int)states.size();
                            if (states[t_j].hasCenter) {
                                float w =  offsetToWeight(j);
                                summ_tv += (float)(-j) * states[t_j].center[c] * w;
                                summ_v += (float)states[t_j].center[c] * w ;
                                summ_v2 += (float)states[t_j].center[c]* states[t_j].center[c] * w;
                            }
                        }

                        estA[c] = (total_w * summ_tv - summ_t * summ_v) / (total_w * summ_t2 - summ_t*summ_t);
                        estB[c] = (summ_v - estA[c] * summ_t) / total_w;
                    }

                    hasLinearEstimate = true;
                } else if (n == 1) {
                    estMeterSize /= total_w;
                    estA[0] = 0.0f;
                    estA[1] = 0.0f;
                    hasLinearEstimate = true;
                } else {
                    hasLinearEstimate = false;
                }
            }

            Track(TrackerConfig* config) : config(config), states(config->trkSize) {
                for (auto& s : states) {
                    s.js.resize(config->jointCount);
                }
            }
        };
    private:
        TrackerConfig config;

        int trk_ids = 0;
        int trk_uids = 0;
        int cur_frame = 0;

#ifndef BM_TRACKER_DISABLE_FEATURES
        std::map< int, FeatureList > pose_features;
#endif 
        std::vector< Track> tracklets;

        static float intersect_segments(float a1, float b1, float a2, float b2) {
            if (a1 <= a2) {
                if (b1 <= a2) return 0.0f;
                if (b1 >= b2) return b2 - a2;
                return b1 - a2;
            } else {
                return intersect_segments(a2, b2, a1, b1);
            }
        }

        float calc_score(PoseData& pose, Track& track) {
            if (track.trkSize == 0 || !track.hasLinearEstimate) {
                return -config.cNew;
            } else {
                float score = 0.0f;

                float bs = std::min(track.estMeterSize, (float) pose.baseRadius);
                if (config.cCenter != 0.0f) {
                    float est_x = track.estA[0] * (cur_frame - track.estT) + track.estB[0];
                    float est_y = track.estA[1] * (cur_frame - track.estT) + track.estB[1];

                    float cdx = est_x - (float)pose.centerX;
                    float cdy = est_y - (float)pose.centerY;
                    float s2 = cdx*cdx + cdy*cdy;
                    
                    score -= sqrtf(s2) / bs * config.cCenter;
                }
                
                if (config.cPose != 0.0f) {
                    int n_paired = 0;
                    float s_paired = 0.0f;
                    for (int limb = 0; limb < config.jointCount; limb++) {
                        if (pose.joint[limb].present) {
                            for (int j = 0; j < std::min(track.trkSize, config.maxLimbDelay); j++) {
                                int t_j = track.curI - 1 - j;
                                while (t_j < 0) t_j += config.trkSize;
                                if (track.states[t_j].js[limb].present) {
                                    float dx = track.states[t_j].js[limb].c[0] - (float)pose.joint[limb].x;
                                    float dy = track.states[t_j].js[limb].c[1] - (float)pose.joint[limb].y;
                                    float s2 = (dx*dx + dy*dy);
                                    float d = sqrtf(s2) / bs / config.cPoseMaxR;
                                    if (d > 1.0f) {
                                        d = 1.0f;
                                    }
                                    s_paired += d;
                                    n_paired++;
                                    break;
                                }
                            }
                        } 
                    }

                    if (n_paired == 0) {
                        score -= config.cPose;
                    } else {
                        s_paired /= n_paired;
                        score -= config.cPose * s_paired;
                    }
                }

                if (config.cPckh != 0.0f) {
                    int n_paired = 0;
                    float s_paired = 0.0f;
                    for (int j = 0; j < std::min(track.trkSize, config.maxLimbDelay); j++) {
                        int t_j = track.curI - 1 - j;
                        while (t_j < 0) t_j += config.trkSize;
                        if (track.states[t_j].present) {
                            for (int limb = 0; limb < config.jointCount; limb++) {
                                if (pose.joint[limb].present && track.states[t_j].js[limb].present) {
                                    float dx = track.states[t_j].js[limb].c[0] - (float)pose.joint[limb].x;
                                    float dy = track.states[t_j].js[limb].c[1] - (float)pose.joint[limb].y;
                                    float s2 = (dx*dx + dy*dy);
                                    if (s2 > (bs * config.cPckhMaxR) * (bs * config.cPckhMaxR)) {
                                        s_paired += 1.0f;
                                    }
                                    n_paired++;
                                }
                            } 
                            break;
                        }
                    }

                    if (n_paired == 0) {
                        score -= config.cPckh;
                    } else {
                        s_paired /= n_paired;
                        score -= config.cPckh * s_paired;
                    }
                }

                if (config.cIOU != 0.0f) {
                    for (int j = 0; j < std::min(track.trkSize, config.trkSize); j++) {
                        int t_j = track.curI - 1 - j;
                        while (t_j < 0) t_j += config.trkSize;
                        if (track.states[t_j].present) {
                            float min_x = 99999.0f;
                            float min_y = min_x;
                            float max_x = -min_x;
                            float max_y = -min_x;
                            for (int limb = 0; limb < config.jointCount; limb++) {
                                if (track.states[t_j].js[limb].present) {
                                    min_x = std::min(min_x, (float) track.states[t_j].js[limb].c[0]);
                                    max_x = std::max(max_x, (float) track.states[t_j].js[limb].c[0]);
                                    min_y = std::min(min_y, (float) track.states[t_j].js[limb].c[1]);
                                    max_y = std::max(max_y, (float) track.states[t_j].js[limb].c[1]);
                                }
                            }
                            
                            int iw = intersect_segments(min_x, max_x, pose.bboxMin[0],  pose.bboxMax[0]);
                            int ih = intersect_segments(min_y, max_y, pose.bboxMin[1],  pose.bboxMax[1]);
                            float ii = (float)(iw * ih);
                            float iou = ii / ((max_x - min_x) * (max_y - min_y) +
                                (pose.bboxMax[0] - pose.bboxMin[0]) * (pose.bboxMax[1] - pose.bboxMin[1]) - ii);
                            score -= (1.0f - iou) * config.cIOU;
                            break;
                        }
                    }
                }

#ifndef BM_TRACKER_DISABLE_FEATURES
                if (config.cColor != 0.0f) {
                    score -= config.cColor * (1.0f - track.fTrack.matchWith(pose_features[pose.id], config.clrBeta,
                        config.clrUseSqr));
                }
#endif

                return score;
            }
        
        }

        //Vengerian alogirmth implementation from acm.mipt.ru
        std::vector< std::pair < int, int > > run_vengerian(const std::vector< std::vector < int > > & m) {
            // matrix sizes
            const int inf = std::numeric_limits<int>::max();
            if (!m.size()) {
                return {};
            }
            int height = m.size(), width = m[0].size();
            
            // subtract values for rows(u) and columns (v)
            std::vector<float> u(height, 0), v(width, 0);
            
            std::vector<int> markIndices(width, -1);
            
            for(int i = 0; i < height; i++) {
                std::vector<int>  links(width, -1);
                std::vector<int>  mins(width, inf);
                std::vector<int>  visited(width, 0);
                
                int markedI = i, markedJ = -1, j;
                while(markedI != -1) {
                    j = -1;
                    for(int j1 = 0; j1 < width; j1++)
                        if(!visited[j1]) {
                        if(m[markedI][j1] - u[markedI] - v[j1] < mins[j1]) {
                            mins[j1] = m[markedI][j1] - u[markedI] - v[j1];
                            links[j1] = markedJ;
                        }
                        if(j==-1 || mins[j1] < mins[j])
                            j = j1;
                        }
                        
                    int delta = mins[j];
                    for(int j1 = 0; j1 < width; j1++)
                        if(visited[j1]) {
                            u[markIndices[j1]] += delta;
                            v[j1] -= delta;
                        } else {
                            mins[j1] -= delta;
                        }
                    u[i] += delta;
                    
                    visited[j] = 1;
                    markedJ = j;
                    markedI = markIndices[j];   
                }
                
                for(; links[j] != -1; j = links[j])
                    markIndices[j] = markIndices[links[j]];
                markIndices[j] = i;
            }
            
            std::vector< std::pair < int, int >>  result;
            for(int j = 0; j < width; j++)
                if(markIndices[j] != -1)
                    result.push_back({markIndices[j], j});
            return result;
        }


        std::vector< std::vector<float> > precalc_scores;

        float get_score (std::vector<PoseData*> &poses, int pose_id, int track_id) {
            if (tracklets[track_id].trkSize == 0) {
                return calc_score(*poses[pose_id], tracklets[track_id]);
            } else {
                return precalc_scores[pose_id][track_id];
            }
        }

        int get_int_score(std::vector<PoseData*> &poses, int pose_id, int track_id) {
            float s = -get_score(poses, pose_id, track_id);
            int ins = (int)(s * 2000.0f);
            return ins;
        }

        float try_associate(std::vector<PoseData*> &poses, int new_tracks, bool preserve) {
            for (int i = 0; i < new_tracks; i++) {
                Track t{&config};
                t.lastFrame =  -1;
                t.id = trk_ids++;
                tracklets.push_back(t);
            }

            assert(tracklets.size() >= poses.size());

            //std::shuffle(poses.begin(), poses.end(), urng);

            int n = 0;
            for (auto &i : tracklets) {
                i.curPose = -1;
            }
            for (auto i : poses) {
                i->trklId = -1;
            }

            std::vector< std::vector< int > > scores;
            for (int i = 0; i < (int)poses.size(); i++) {
                scores.push_back({});
                scores.back().reserve(tracklets.size());
                for (int j = 0; j < (int)tracklets.size(); j++) {
                    scores[i].push_back(get_int_score(poses, i, j));
                }
            }

            auto r = run_vengerian(scores);

            float score = 0.0f;

            for (auto& p : r) {
                score += get_score(poses, p.first, p.second);
                poses[p.first]->trklId = p.second;
                tracklets[p.second].curPose = p.first;
            }


            if (!preserve) {
                for (int i = 0; i < new_tracks; i++) {
                    tracklets.pop_back();
                }
                trk_ids -= new_tracks;
            } 
            return score;
        }


        void update_tracks( std::vector<PoseData*> &poses) {


            for (auto p : poses) {
                if (p->trklId >= 0) {
                    int trk_id = tracklets[p->trklId].trkUid;
                    if (trk_id < 0) {
                        trk_id = trk_uids++;
                        //printf("New track uid: %d\n", trk_id);
                        tracklets[p->trklId].trkUid = trk_id;
                    }
                    p->track = trk_id;
                }
            }
            for (auto &t : tracklets) {
                auto& st = t.states[t.curI];
                if (t.curPose >= 0) {
                    st.present = true;
                    auto& p = *poses[t.curPose];
                    st.meterSize = p.baseRadius;

                    float c_x = 0.0f, c_y = 0.0f, c_w = 0.0f;

                    for( int i = 0; i < config.jointCount; i++) {
                        if (p.joint[i].present) {
                            st.js[i].c[0] = p.joint[i].x;
                            st.js[i].c[1] = p.joint[i].y;
                            st.js[i].present = true;
                        } else {
                            st.js[i].present = false;
                        }
                    } 
                    st.center[0] = p.centerX;
                    st.center[1] = p.centerY;
                    st.hasCenter = true;

                    t.lastFrame = cur_frame;
                    t.missingCount = 0;

#ifndef BM_TRACKER_DISABLE_FEATURES
                    if (config.cColor != 0.0f) {
                        t.fTrack.update(pose_features[p.id], config.clrAlpha);
                    }
#endif
                } else {
                    st.present = true;
                    st.hasCenter = true;
                    t.missingCount++;
                    t.expired = t.missingCount >= (config.trkSize - 1);
                }

                t.curI++;
                if (t.curI == config.trkSize)
                    t.curI = 0;
                t.trkSize++;

                if (t.curPose >= 0)
                    t.update_estimate(cur_frame);
            }

            
            auto new_end = std::remove_if(tracklets.begin(), tracklets.end(),
                                [](const Track& t)
                                { return t.expired || (t.trkUid < 0); });
            tracklets.erase(new_end, tracklets.end());
        }

    public:

        void process(Frame& frame, matType m = {}) {
            
            #ifndef BM_TRACKER_DISABLE_FEATURES
            FeatureExtractor fEx;
            pose_features.clear();
            #endif

            std::vector< PoseData* > poses;
            cur_frame++;

            int n = 0;
            for (auto &i : frame.poses) {
                auto& cj = i.joint[config.centerJoint];
                i.track = -1;
                i.calcBbox();
                if (cj.present) {
                    i.centerX = cj.x;
                    i.centerY = cj.y;
                } else {
                    if (config.alwaysDeriveCenter) {
                        i.centerX = (i.bboxMax[0] + i.bboxMin[0]) * 0.5f;
                        i.centerY = i.bboxMax[0] * 0.8f + i.bboxMax[1] * 0.2f;
                    } else {
                        continue;
                    }
                }
                i.baseRadius = std::max(30.0, std::max(i.bboxMax[1] - i.bboxMin[1], i.bboxMax[0] - i.bboxMin[0]) * 0.6);
                i.trklId = -1;
                i.id = n++;
                poses.push_back(&i);
                
                #ifndef BM_TRACKER_DISABLE_FEATURES
                if (config.cColor != 0.0f) {
                    pose_features[i.id] = fEx.calcFeatures(m, i);
                }
                #endif
            }
            

            precalc_scores.resize(poses.size());
            for (int i = 0; i < (int)poses.size(); i++) {
                precalc_scores[i].resize(tracklets.size());
                for (int j = 0; j < (int)tracklets.size(); j++) {
                    precalc_scores[i][j] = calc_score(*poses[i], tracklets[j]);
                }
            }

            int c_new = std::max(0, (int) poses.size() - (int) tracklets.size());
            
            float c_score = try_associate(poses, c_new, false);
            while (true) {
                float n_score = try_associate(poses, c_new + 1, false);
                //printf("Score %.2f -> %.2f (%d, %d)\n", c_score, n_score, (int)(tracklets.size() + c_new), trk_ids);
                if (n_score > c_score + 0.002f) {
                    c_new++;
                    c_score = n_score;
                } else {
                    break;
                }
            }

            try_associate(poses, c_new, true);
            update_tracks(poses);
            precalc_scores.clear();

        }
        
        Tracker(const TrackerConfig& config) : config(config) {

        }
    };

}