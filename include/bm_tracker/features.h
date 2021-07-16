#pragma once

#include "frame.h"

#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>

namespace BM_Tracker {

    typedef cv::Mat matType; 

    struct FeatureValue {
        float v;
        bool present = false;

        void operator =(float x) {
            present = true;
            v = x;
        }
    };

    struct FeatureList {
        std::vector< FeatureValue > f;
    };

    class FeatureExtractor {
    private:

        struct EllipseFeatureDef {
            int src, trg;
            float w1, w2;
            EllipseFeatureDef( int src, int trg, float w1, float w2 = -1.0f) : src(src), trg(trg), w1(w1), w2(w2) {
                if (w2 == -1.0f) w2 = w1;
            }

        };
        std::vector< EllipseFeatureDef > & getDefs() {
            static std::vector< EllipseFeatureDef > v = {
                {1, 8, 0.3f, 0.2f},
                {2, 5, 0.2f, 0.2f},
                {5, 6, 0.2f, 0.2f},
                {6, 7, 0.15f, 0.15f},
                {2, 3, 0.2f, 0.2f},
                {3, 4, 0.15f, 0.15f},
                {9, 12, 0.3f, 0.3f},
                {9, 10, 0.2f, 0.2f},
                {10, 11, 0.2f, 0.15f},
                {12, 13, 0.2f, 0.2f},
                {13, 14, 0.2f, 0.15f},
                {0, 1, 0.4f, 0.2f},
                {17, 18, 0.4f, 0.4f}
            };
            return v;
        }

        struct EllipseResult {
            float avg[3] = {};
            float avg2[3] = {};
        };


        EllipseResult calculateEllipse(cv::Mat& m, cv::Point2i p1, cv::Point2i p2, float hw1, float hw2, bool ellipse = false, cv::Scalar color = {}) {
            EllipseResult r;

            if (p1 == p2)
                return r;
            float cw = 0.0f;

            float mhw = std::max(hw1, hw2);

            cv::Rect2i calcr(p1, p2);
            calcr.x -= (int)mhw;
            calcr.y -= (int)mhw;
            calcr.width += 2 * (int)mhw;
            calcr.height += 2 * (int)mhw;

            //cv::Mat& m = frame.frame->m;
            cv::Rect matr(0, 0, m.cols, m.rows);
            calcr &= matr;

            if (calcr.empty())
                return r;
            //printf("calcr %d %d %d %d, chans %d\n", calcr.x, calcr.y, calcr.width, calcr.height, m.channels());

            cv::Vec2f mainAxis { (float) (p2 - p1).x, (float) (p2 - p1).y };
            cv::Vec2f secAxis { mainAxis[1], -mainAxis[0]};

            mainAxis /= mainAxis[0] * mainAxis[0] + mainAxis[1] * mainAxis[1];
            mainAxis *= 2.0f;

            secAxis /= sqrtf(secAxis[0] * secAxis[0] + secAxis[1] * secAxis[1]);
            secAxis /= hw1;
            cv::Point2i cp = (p1 + p2) / 2;

            float hwCoef = hw2 / hw1;
            float dy = (float)( calcr.y - cp.y);
            for (int y = 0; y < calcr.height; y++, dy += 1.0f) {
                uint8_t* d = m.ptr<uint8_t>(y + calcr.y);
                d += m.channels() * calcr.x;
                float dx = (float)(calcr.x - cp.x);
                for (int x = 0; x < calcr.width; x++, dx += 1.0f) {
                    
                    float pm = mainAxis[0] * dx + mainAxis[1] * dy;
                    float ps = secAxis[0] * dx + secAxis[1] *  dy;
                    bool pass = false;
                    if (ellipse) {
                        pm *= pm;
                        ps *= ps;
                        if (pm + ps < 1.0f) {
                            pass = true;
                        }
                    } else {
                        if (std::abs(pm) < 1.0f) {
                            float a = (pm + 1.0f) * 0.5f;
                            float cps = (1-a) + a * hwCoef;
                            pass = std::abs(ps) < cps;
                        }
                    }
                    if (pass) {
                        for (int c = 0; c < 3; c++) {
                            float v = (float) *(d+c);
                            r.avg[c] += v;
                            r.avg2[c] += v* v;
                            cw += 1.0f;
                        }
                        if (debug_out) {
                            *d = color[0] / 2+ *d / 2;
                            *(d+1) = color[1] / 2+ *(d+1) / 2;
                            *(d+2) = color[2] / 2+ *(d+2) / 2;
                        }
                    }
                    d += 3;
                }
            }

            if (cw == 0.0f)
                return r;
            
            for (int c = 0; c < 3; c++) {
                r.avg[c] /= cw * 255.0f;
                r.avg2[c] /= cw * (255.0f * 255.0f);
            }

            return r;
        }

    public:
        bool debug_out = false;

        FeatureList calcFeatures(cv::Mat& m, PoseData& pd, const std::vector<cv::Scalar> &colors
            = {}) {
            auto& defs = getDefs();
            FeatureList feats;
            feats.f.resize(defs.size() * 3 * 2);
            int n = 0;
            for (auto &d : defs) {
                auto & js = pd.joint[d.src];
                auto & jt = pd.joint[d.trg];
                if (js.present && jt.present) {
                    float dx = js.x - jt.x;
                    float dy = js.y - jt.y;
                    float baseLen = sqrtf(dx*dx+dy*dy) + 0.01f;
                    EllipseResult er = calculateEllipse(m, {js.x, js.y}, {jt.x, jt.y}, 
                        baseLen * d.w1, baseLen * d.w2, false, colors.size() ? colors[n % colors.size()] : cv::Scalar());
                    for (int i = 0; i < 3; i++) {
                        feats.f[n*6+i] = er.avg[i];
                        feats.f[n*6+3+i] = er.avg2[i];
                    }
                } 
                n++;
            }
            return feats;
        }

        int feature_count() {
            return getDefs().size() * 3 * 2;
        }

    };

    struct FeatureTrack {
        struct Stat {
            float avg = 0.5f, avg2 = 0.5f, w = 1.0f;
        };
        std::vector< Stat > stats;

        void update(const FeatureList& fl, float alpha = 0.8f) {
            if (!stats.size()) {
                stats.resize(fl.f.size());
            }
            assert(stats.size() == fl.f.size());

            int n = 0;
            for (auto& f : fl.f) {
                if (f.present) {
                    auto& s = stats[n];
                    
                    float nw = s.w + 1.0f;

                    s.avg = (s.avg * s.w + f.v) / nw;
                    s.avg2 = (s.avg2 * s.w + f.v * f.v) / nw;
                    s.w = nw * alpha;
                } else {
                    auto& s = stats[n];
                    s.w *= alpha;
                }
                n++;
            }
        }

        //1.0f - perfect match; 0.0f - bad match
        float matchWith(FeatureList& fl, float beta = 1.5f, bool useSqr = false) {
            int n = 0;
            int total = 0;
            float score = 0.0f;
            float rbeta = 1.0f / beta;
            for (auto& f : fl.f) {
                if (f.present) {
                    auto& s = stats[n];
                    float delta = fabsf(f.v - s.avg);
                    float sigma = sqrtf(std::max(0.0001f, s.avg2 - s.avg * s.avg));
                    float ns = delta / sigma;
                    if (ns < beta) {
                        if (useSqr) {
                            score += 1.0f - (ns * rbeta) * (ns * rbeta);
                        } else {
                            score += 1.0f - ns * rbeta;
                        }
                    }
                    total++;
                }
                n++;
            }
            if (total > 0) {
                score /= total;
            }
            return score;
        }
    };

}