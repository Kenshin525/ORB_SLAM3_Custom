#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
#include <string>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <librealsense2/rs.hpp>
#include <librealsense2/rsutil.h>
#include "Config.h"
#include "System.h"

using Clock = std::chrono::steady_clock;

static std::atomic<bool> running(true);
static void on_signal(int) { running = false; }

static double ms_between(const Clock::time_point& a, const Clock::time_point& b)
{
    return std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(b - a).count();
}

static std::string num(double v, int p = 10)
{
    std::ostringstream s;
    s << std::fixed << std::setprecision(p) << v;
    return s.str();
}

static bool key_match(const std::string& line, const std::string& key)
{
    size_t i = line.find_first_not_of(" \t");
    if (i == std::string::npos || line.compare(i, key.size(), key) != 0)
        return false;

    i += key.size();
    while (i < line.size() && (line[i] == ' ' || line[i] == '\t'))
        ++i;

    return i < line.size() && line[i] == ':';
}

static float physical_baseline_m(const rs2::device& dev)
{
    for (auto s : dev.query_sensors())
    {
        try
        {
            if (s.supports(RS2_OPTION_STEREO_BASELINE))
                return s.get_option(RS2_OPTION_STEREO_BASELINE) * 0.001f;
        }
        catch (...) {}
    }
    return -1.0f;
}

static const char* distortion_name(rs2_distortion m)
{
    switch (m)
    {
        case RS2_DISTORTION_NONE: return "None";
        case RS2_DISTORTION_MODIFIED_BROWN_CONRADY: return "Modified Brown-Conrady";
        case RS2_DISTORTION_INVERSE_BROWN_CONRADY: return "Inverse Brown-Conrady";
        case RS2_DISTORTION_BROWN_CONRADY: return "Brown-Conrady";
        case RS2_DISTORTION_FTHETA: return "F-Theta";
        case RS2_DISTORTION_KANNALA_BRANDT4: return "Kannala-Brandt4";
        default: return "Unknown";
    }
}

static bool read_yaml_scalar(const std::string& path,
                             const std::string& key,
                             double& value)
{
    std::ifstream in(path);
    if (!in)
        return false;

    std::string line;
    while (std::getline(in, line))
    {
        if (!key_match(line, key))
            continue;

        size_t colon = line.find(':');
        if (colon == std::string::npos)
            return false;

        try
        {
            value = std::stod(line.substr(colon + 1));
            return true;
        }
        catch (...)
        {
            return false;
        }
    }
    return false;
}

static void write_runtime_yaml(const std::string& src,
                               const std::string& dst,
                               const rs2_intrinsics& k,
                               float depth_scale,
                               int fps)
{
    std::ifstream in(src);
    if (!in)
        throw std::runtime_error("Cannot open YAML template: " + src);

    if (depth_scale <= 0.0f)
        throw std::runtime_error("Invalid RealSense depth scale");

    std::map<std::string, std::string> v;
    v["Camera1.fx"] = num(k.fx);
    v["Camera1.fy"] = num(k.fy);
    v["Camera1.cx"] = num(k.ppx);
    v["Camera1.cy"] = num(k.ppy);

    // The adapter supplies a rectified/ideal pinhole image to ORB-SLAM3.
    v["Camera1.k1"] = "0.0";
    v["Camera1.k2"] = "0.0";
    v["Camera1.p1"] = "0.0";
    v["Camera1.p2"] = "0.0";

    v["Camera.width"] = std::to_string(k.width);
    v["Camera.height"] = std::to_string(k.height);
    v["Camera.fps"] = std::to_string(fps);
    v["Camera.RGB"] = "1";

    // Raw Z16 depth is converted by ORB-SLAM3 as depth / DepthMapFactor.
    v["RGBD.DepthMapFactor"] = num(1.0 / depth_scale, 6);

    // IMPORTANT:
    // Do NOT replace Stereo.b with the RealSense physical IR baseline.
    // In ORB-SLAM3 RGB-D, Stereo.b is also used to construct a virtual
    // right-image coordinate. Keep the explicitly chosen template value.

    std::ofstream out(dst);
    if (!out)
        throw std::runtime_error("Cannot write runtime YAML: " + dst);

    std::string line;
    while (std::getline(in, line))
    {
        bool done = false;
        for (const auto& kv : v)
        {
            if (key_match(line, kv.first))
            {
                out << kv.first << ": " << kv.second << "\n";
                done = true;
                break;
            }
        }
        if (!done)
            out << line << "\n";
    }
}

static void make_maps(const rs2_intrinsics& k,
                      cv::Mat& mx,
                      cv::Mat& my,
                      bool& use)
{
    bool coeffs_zero = true;
    for (int i = 0; i < 5; ++i)
        coeffs_zero = coeffs_zero && std::fabs(k.coeffs[i]) < 1e-10f;

    use = (k.model != RS2_DISTORTION_NONE && !coeffs_zero);
    if (!use)
        return;

    mx.create(k.height, k.width, CV_32FC1);
    my.create(k.height, k.width, CV_32FC1);

    // Destination image: ideal pinhole with the active stream's fx/fy/cx/cy.
    // cv::remap needs destination->source coordinates.
    for (int y = 0; y < k.height; ++y)
    {
        float* px = mx.ptr<float>(y);
        float* py = my.ptr<float>(y);

        for (int x = 0; x < k.width; ++x)
        {
            float p[3] = {
                (static_cast<float>(x) - k.ppx) / k.fx,
                (static_cast<float>(y) - k.ppy) / k.fy,
                1.0f
            };
            float q[2];
            rs2_project_point_to_pixel(q, &k, p);
            px[x] = q[0];
            py[x] = q[1];
        }
    }
}

struct TimingStats
{
    uint64_t frames = 0;
    uint64_t frame_gaps = 0;

    double wait_sum = 0.0;
    double align_sum = 0.0;
    double remap_sum = 0.0;
    double track_sum = 0.0;
    double process_sum = 0.0;

    double wait_max = 0.0;
    double align_max = 0.0;
    double remap_max = 0.0;
    double track_max = 0.0;
    double process_max = 0.0;

    uint64_t last_frame_number = 0;
    bool have_last_frame = false;

    void add(double wait_ms,
             double align_ms,
             double remap_ms,
             double track_ms,
             uint64_t frame_number)
    {
        ++frames;

        wait_sum += wait_ms;
        align_sum += align_ms;
        remap_sum += remap_ms;
        track_sum += track_ms;

        const double process_ms = align_ms + remap_ms + track_ms;
        process_sum += process_ms;

        wait_max = std::max(wait_max, wait_ms);
        align_max = std::max(align_max, align_ms);
        remap_max = std::max(remap_max, remap_ms);
        track_max = std::max(track_max, track_ms);
        process_max = std::max(process_max, process_ms);

        if (have_last_frame && frame_number > last_frame_number + 1)
            frame_gaps += frame_number - last_frame_number - 1;

        last_frame_number = frame_number;
        have_last_frame = true;
    }

    void print_window(uint64_t window_start,
                      double w_wait,
                      double w_align,
                      double w_remap,
                      double w_track,
                      double w_process,
                      uint64_t w_gaps,
                      uint64_t window_frames) const
    {
        if (window_frames == 0)
            return;

        const double inv = 1.0 / static_cast<double>(window_frames);
        const double avg_process = w_process * inv;
        const double compute_fps = avg_process > 0.0 ? 1000.0 / avg_process : 0.0;

        std::cout << std::fixed << std::setprecision(2)
                  << "[perf frames " << window_start << "-"
                  << (window_start + window_frames - 1) << "] "
                  << "wait=" << w_wait * inv << "ms, "
                  << "align=" << w_align * inv << "ms, "
                  << "remap=" << w_remap * inv << "ms, "
                  << "TrackRGBD=" << w_track * inv << "ms, "
                  << "process=" << avg_process << "ms"
                  << " (~" << compute_fps << " FPS), "
                  << "camera_frame_gaps=" << w_gaps
                  << "\n";
    }

    void print_final() const
    {
        if (frames == 0)
            return;

        const double inv = 1.0 / static_cast<double>(frames);
        const double avg_process = process_sum * inv;
        const double compute_fps = avg_process > 0.0 ? 1000.0 / avg_process : 0.0;

        std::cout << "\n========== RealSense/SLAM timing ==========\n"
                  << std::fixed << std::setprecision(2)
                  << "Frames:             " << frames << "\n"
                  << "Camera frame gaps:  " << frame_gaps << "\n"
                  << "wait_for_frames:     " << wait_sum * inv
                  << " ms avg / " << wait_max << " ms max\n"
                  << "align:               " << align_sum * inv
                  << " ms avg / " << align_max << " ms max\n"
                  << "remap RGB+Depth:     " << remap_sum * inv
                  << " ms avg / " << remap_max << " ms max\n"
                  << "TrackRGBD:           " << track_sum * inv
                  << " ms avg / " << track_max << " ms max\n"
                  << "processing total:    " << avg_process
                  << " ms avg / " << process_max << " ms max\n"
                  << "compute throughput:  " << compute_fps << " FPS\n"
                  << "===========================================\n";
    }
};

int main(int argc, char** argv)
{
    if (argc < 3 || argc > 4)
    {
        std::cerr << "Usage: " << argv[0]
                  << " vocabulary template_yaml [trajectory_prefix]\n";
        return 1;
    }

    std::signal(SIGINT, on_signal);
    std::signal(SIGTERM, on_signal);

    try
    {
        rs2::context ctx;
        auto devs = ctx.query_devices();
        if (devs.size() == 0)
            throw std::runtime_error("No RealSense device connected");

        rs2::device dev = devs[0];

        const std::string name =
            dev.supports(RS2_CAMERA_INFO_NAME)
                ? dev.get_info(RS2_CAMERA_INFO_NAME)
                : "RealSense";

        const std::string serial =
            dev.supports(RS2_CAMERA_INFO_SERIAL_NUMBER)
                ? dev.get_info(RS2_CAMERA_INFO_SERIAL_NUMBER)
                : "unknown";

        std::cout << "Device: " << name << "\n"
                  << "Serial: " << serial << "\n";

        const int W = 640;
        const int H = 480;
        const int FPS = 30;

        rs2::pipeline pipe;
        rs2::config cfg;
        cfg.enable_stream(RS2_STREAM_COLOR, W, H, RS2_FORMAT_RGB8, FPS);
        cfg.enable_stream(RS2_STREAM_DEPTH, W, H, RS2_FORMAT_Z16, FPS);

        auto profile = pipe.start(cfg);

        auto cp = profile.get_stream(RS2_STREAM_COLOR)
                          .as<rs2::video_stream_profile>();

        rs2_intrinsics k = cp.get_intrinsics();

        auto depth_sensor =
            profile.get_device().first<rs2::depth_sensor>();

        const float depth_scale = depth_sensor.get_depth_scale();
        const float physical_b = physical_baseline_m(profile.get_device());

        double template_b = std::numeric_limits<double>::quiet_NaN();
        const bool have_template_b =
            read_yaml_scalar(argv[2], "Stereo.b", template_b);

        std::cout << "\nActive calibration\n"
                  << " fx/fy: " << k.fx << " / " << k.fy << "\n"
                  << " cx/cy: " << k.ppx << " / " << k.ppy << "\n"
                  << " size : " << k.width << " x " << k.height << "\n"
                  << " model: " << distortion_name(k.model) << "\n"
                  << " coeff: "
                  << k.coeffs[0] << ", "
                  << k.coeffs[1] << ", "
                  << k.coeffs[2] << ", "
                  << k.coeffs[3] << ", "
                  << k.coeffs[4] << "\n"
                  << " depth scale: " << depth_scale << " m/unit\n";

        if (physical_b > 0.0f)
            std::cout << " physical IR baseline: " << physical_b << " m\n";

        if (have_template_b)
            std::cout << " ORB-SLAM3 RGB-D Stereo.b: "
                      << template_b
                      << " m (kept from template; not auto-overwritten)\n";

        cv::Mat mx, my;
        bool remap = false;

        std::cout << "Building rectification map..." << std::flush;
        make_maps(k, mx, my, remap);
        std::cout << (remap ? " enabled\n" : " not needed\n");

        const std::string runtime_yaml =
            "Examples/RGB-D/RealSense_D400_runtime_v2.yaml";

        write_runtime_yaml(argv[2], runtime_yaml, k, depth_scale, FPS);

        std::cout << "Runtime YAML: " << runtime_yaml << "\n";

        rs2::align align(RS2_STREAM_COLOR);

        ORB_SLAM3::System SLAM(
            argv[1],
            runtime_yaml,
            ORB_SLAM3::System::RGBD,
            true
        );

        std::cout << "\nTracking started. Ctrl+C to stop.\n"
                  << "A performance line will be printed every 60 frames.\n";

        TimingStats stats;

        double w_wait = 0.0;
        double w_align = 0.0;
        double w_remap = 0.0;
        double w_track = 0.0;
        double w_process = 0.0;
        uint64_t w_gaps = 0;
        uint64_t w_frames = 0;
        uint64_t window_start = 1;
        uint64_t prev_frame_number = 0;
        bool have_prev_frame_number = false;

        while (running && !SLAM.isShutDown())
        {
            const auto t0 = Clock::now();
            auto frames = pipe.wait_for_frames();
            const auto t1 = Clock::now();

            auto aligned = align.process(frames);
            const auto t2 = Clock::now();

            auto color = aligned.get_color_frame();
            auto depth = aligned.get_depth_frame();
            if (!color || !depth)
                continue;

            cv::Mat c0(
                cv::Size(k.width, k.height),
                CV_8UC3,
                const_cast<void*>(color.get_data()),
                cv::Mat::AUTO_STEP
            );

            cv::Mat d0(
                cv::Size(k.width, k.height),
                CV_16UC1,
                const_cast<void*>(depth.get_data()),
                cv::Mat::AUTO_STEP
            );

            cv::Mat c, d;

            if (remap)
            {
                cv::remap(
                    c0, c, mx, my,
                    cv::INTER_LINEAR,
                    cv::BORDER_CONSTANT
                );

                // Metric depth must not be bilinearly interpolated.
                cv::remap(
                    d0, d, mx, my,
                    cv::INTER_NEAREST,
                    cv::BORDER_CONSTANT
                );
            }
            else
            {
                c = c0;
                d = d0;
            }

            const auto t3 = Clock::now();

            SLAM.TrackRGBD(
                c,
                d,
                color.get_timestamp() * 1e-3
            );

            const auto t4 = Clock::now();

            const double wait_ms = ms_between(t0, t1);
            const double align_ms = ms_between(t1, t2);
            const double remap_ms = ms_between(t2, t3);
            const double track_ms = ms_between(t3, t4);
            const double process_ms = align_ms + remap_ms + track_ms;

            const uint64_t frame_number = color.get_frame_number();

            uint64_t gap = 0;
            if (have_prev_frame_number &&
                frame_number > prev_frame_number + 1)
            {
                gap = frame_number - prev_frame_number - 1;
            }

            prev_frame_number = frame_number;
            have_prev_frame_number = true;

            stats.add(
                wait_ms,
                align_ms,
                remap_ms,
                track_ms,
                frame_number
            );

            w_wait += wait_ms;
            w_align += align_ms;
            w_remap += remap_ms;
            w_track += track_ms;
            w_process += process_ms;
            w_gaps += gap;
            ++w_frames;

            if (w_frames >= 60)
            {
                stats.print_window(
                    window_start,
                    w_wait,
                    w_align,
                    w_remap,
                    w_track,
                    w_process,
                    w_gaps,
                    w_frames
                );

                window_start += w_frames;
                w_wait = w_align = w_remap = w_track = w_process = 0.0;
                w_gaps = 0;
                w_frames = 0;
            }
        }

        pipe.stop();

        stats.print_final();

        std::cout << "\nStopping ORB-SLAM3...\n";
        SLAM.Shutdown();

        const std::string prefix =
            argc == 4 ? argv[3] : "RealSense";

        SLAM.SaveTrajectoryTUM(
            prefix + "_CameraTrajectory_v2.txt"
        );

        SLAM.SaveKeyFrameTrajectoryTUM(
            prefix + "_KeyFrameTrajectory_v2.txt"
        );

        std::cout << "Saved trajectories with prefix: "
                  << prefix << "\n";

#ifdef REGISTER_TIMES
        std::cout << "ORB-SLAM3 REGISTER_TIMES is enabled. "
                  << "Check ExecTimeMean.txt after shutdown.\n";
#else
        std::cout << "WARNING: REGISTER_TIMES is not enabled in include/Config.h.\n";
#endif

        return 0;
    }
    catch (const rs2::error& e)
    {
        std::cerr << "RealSense error: " << e.what() << "\n";
        return 2;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Error: " << e.what() << "\n";
        return 3;
    }
}
