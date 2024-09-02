#include "decoder_node.h"

#include <chrono>
#include <opencv2/opencv.hpp>

#include "signal/signal.h"

extern "C"
{
#include "libavcodec/avcodec.h"
#include "libavformat/avformat.h"
#include "libavutil/avutil.h"
#include "libavutil/frame.h"
#include "libavutil/pixfmt.h"
#include "libswscale/swscale.h"
}

namespace cv_infer
{
bool DecoderNode::Init(const std::string &name)
{
    if (name.empty())
    {
        LOGE("DecoderNode name is empty");
        return false;
    }
    URI = name;
    LOGI("try Open [%s]", name.c_str());
    if (not Open())
    {
        LOGE("Open [%s] failed", name.c_str());
        return false;
    }
    LOGI("Open [%s] success", name.c_str());
    return true;
}

bool DecoderNode::Open()
{
    Ctx = avformat_alloc_context();
    if (Ctx == nullptr)
    {
        return false;
    }
    AVDictionary *format_opts = NULL;

    av_dict_set(&format_opts, "stimeout", "20000",
                0);  // 设置链接超时时间（us）
    av_dict_set(&format_opts, "rtsp_transport", "tcp",
                0);  // 设置推流的方式，默认udp。
    av_dict_set(&format_opts, "timeout", "6000000",
                0);  // 在进行网络操作时允许的最大等待时间。1秒
    av_dict_set(&format_opts, "max_analyze_duration", "10", 0);
    av_dict_set(&format_opts, "probesize", "2048", 0);
    auto                                                      start_time = std::chrono::system_clock::now();
    std::unique_ptr<AVDictionary *, decltype(av_dict_free) *> free_guard{&format_opts, av_dict_free};
    if (auto ret = avformat_open_input(&Ctx, URI.c_str(), nullptr, &format_opts); ret != 0)
    {
        LOGE("call avformat_open_input return [%d], source = [%s]", ret, URI.c_str());
        return false;
    }
    StreamIdx = GetFirstStreamByType(AVMediaType::AVMEDIA_TYPE_VIDEO);
    if (StreamIdx < 0)
    {
        LOGE("find AVMEDIA_TYPE_VIDEO failed");
        return false;
    }

    CCtx = GetAVCodecContext(StreamIdx);
    if (CCtx == nullptr)
    {
        LOGE("call GetAVCodecContext return nullptr");
        return false;
    }
    CCtx->thread_count = ThreadNum;
    CCtx->thread_type  = FF_THREAD_FRAME;
    Pkt                = av_packet_alloc();
    if (Pkt == nullptr)
    {
        LOGE("call av_packet_alloc return nullptr");
        return false;
    }
    Frame = av_frame_alloc();
    if (Frame == nullptr)
    {
        LOGE("call av_frame_alloc return nullptr");
        return false;
    }

    Width  = CCtx->width;
    Height = CCtx->height;
    LOGD("Source [%s] Width = [%d], Height = [%d]", URI.c_str(), Width, Height);

    Sws = sws_alloc_context();
    if (Sws == nullptr)
    {
        LOGE("call sws_alloc_context return nullptr");
        return false;
    }

    return true;
}

std::vector<std::string> DecoderNode::GetDecoderNameByCodecId(const AVCodecID codec_id) const
{
    if (DecodersPriority.contains(codec_id))
    {
        auto decoders = DecodersPriority.at(codec_id);
        return decoders;
    }
    return {};
}

int DecoderNode::GetFirstStreamByType(enum AVMediaType type) const
{
    for (int i = 0; i < Ctx->nb_streams; i++)
    {
        if (Ctx->streams[i]->codecpar->codec_type == type)
        {
            return i;
        }
    }
    return -1;
}

AVCodecContext *DecoderNode::GetAVCodecContext(int idx) const
{
    auto *par = Ctx->streams[idx]->codecpar;
    LOGI("CodecId = [%d], Width = [%d], Height = [%d]", par->codec_id, par->width, par->height);

    const AVCodec *decoder = nullptr;

    auto decoder_name_list = GetDecoderNameByCodecId(par->codec_id);
    for (const auto &decoder_name : decoder_name_list)
    {
        if (decoder = avcodec_find_decoder_by_name(decoder_name.c_str()); decoder == nullptr)
        {
            LOGT("try to find decoder:[%s] failed", decoder_name.c_str());
            continue;
        }
        break;
    }
    if (decoder == nullptr)
    {
        LOGW("find decoder failed! Will try to find a decoder by codec_id");
        decoder = avcodec_find_decoder(par->codec_id);
    }
    if (decoder == nullptr)
    {
        return nullptr;
    }
    auto *cctx = avcodec_alloc_context3(decoder);
    if (cctx == nullptr)
    {
        return nullptr;
    }
    if (avcodec_parameters_to_context(cctx, par) != 0 or avcodec_open2(cctx, decoder, nullptr) != 0)
    {
        avcodec_free_context(&cctx);
        return nullptr;
    }
    return cctx;
}

bool DecoderNode::Close()
{
    if (Frame != nullptr)
    {
        av_frame_free(&Frame);
        Frame = nullptr;
    }
    if (Pkt != nullptr)
    {
        av_packet_free(&Pkt);
        Pkt = nullptr;
    }
    if (CCtx != nullptr)
    {
        avcodec_free_context(&CCtx);
        CCtx = nullptr;
    }
    if (Ctx != nullptr)
    {
        avformat_close_input(&Ctx);
        Ctx = nullptr;
    }
    return true;
}

cv::Mat DecoderNode::GetOneFrame()
{
    while (av_read_frame(Ctx, Pkt) == 0)
    {
        std::unique_ptr<AVPacket, decltype(av_packet_unref) *> unref_guard{Pkt, av_packet_unref};
        if (Pkt->stream_index != StreamIdx)
        {
            continue;
        }
        if (auto err_code = avcodec_send_packet(CCtx, Pkt); err_code != 0)
        {
            LOGT("can't send packet to decoder, return [%d]", err_code);
            continue;
        }
        if (auto err_code = avcodec_receive_frame(CCtx, Frame); err_code != 0)
        {
            if (err_code == AVERROR(EAGAIN))
            {
                continue;
            }
            LOGT("can't recevie farme from decoder, return [%d]", err_code);
        }
        if (auto frame = DecodeToFrame(Frame); not frame.empty())
        {
            return frame;
        }
    }
    return TryFlushing();
}

cv::Mat DecoderNode::DecodeToFrame(AVFrame *frame)
{
    // sws to convert
    Sws =
        sws_getCachedContext(Sws, Frame->width, Frame->height, static_cast<AVPixelFormat>(Frame->format), Frame->width,
                             Frame->height, AVPixelFormat::AV_PIX_FMT_BGR24, OutFlags, nullptr, nullptr, nullptr);
    cv::Mat image(Frame->height, Frame->width, CV_8UC3);
    int     linesizes[1]{};
    linesizes[0] = image.step1();
    sws_scale(Sws, Frame->data, Frame->linesize, 0, Frame->height, &image.data, linesizes);

    return image;
}

cv::Mat DecoderNode::TryFlushing()
{
    if (not NeedFlushing)
    {
        return cv::Mat();
    }
    avcodec_send_packet(CCtx, NULL);
    auto ret = avcodec_receive_frame(CCtx, Frame);
    if (ret == AVERROR_EOF)
    {
        NeedFlushing = false;
        VideoEOF     = true;
        return cv::Mat();
    }
    if (ret == 0)
    {
        if (auto frame = DecodeToFrame(Frame); not frame.empty())
        {
            return frame;
        }
    }
    return cv::Mat();
}

bool DecoderNode::Run()
{
    while (Running)
    {
        CostTimer.StartTimer();
        auto frame = GetOneFrame();
        CostTimer.EndTimer();
        if (frame.empty())
        {
            LOGW("GetOneFrame return empty frame, exit !!");
            break;
        }
        auto signal      = std::make_shared<SignalImageBGR>(frame);
        signal->FrameIdx = FrameIndex++;
        signal->TimeStamps.push_back(std::chrono::steady_clock::now());

        OutputList[0]->Push(signal);
    }
    return true;
}

}  // namespace cv_infer