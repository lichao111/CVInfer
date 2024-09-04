#include "encoder_node.h"

#include <chrono>
#include <thread>

#include "signal/signal.h"
#include "tools/logger.h"
namespace cv_infer
{
EncoderNode ::~EncoderNode()
{
    if (not Close()) LOGE("Close failed");
}
bool EncoderNode::Init(const std::string &file_name)
{
    OutCfg cfg;
    cfg.out_url = file_name;
    StartTime   = std::chrono::steady_clock::now();
    return Open(cfg);
}
bool EncoderNode::Open(const OutCfg &cfg)
{
    if (auto ret = avformat_alloc_output_context2(&Ctx, nullptr, nullptr, cfg.out_url.c_str()); ret < 0)
    {
        LOGE("avformat_alloc_output_context2 return [%d]", ret);
        return false;
    }

    auto codec = avcodec_find_encoder_by_name(cfg.codec.c_str());
    if (codec == nullptr)
    {
        LOGE("avcodec_find_encoder error");
        return false;
    }

    if (St = avformat_new_stream(Ctx, nullptr); St == nullptr)
    {
        LOGE("avformat_new_stream return nullptr");
        return false;
    }
    St->id = Ctx->nb_streams - 1;
    if (CCtx = avcodec_alloc_context3(codec); CCtx == nullptr)
    {
        LOGE("avcodec_alloc_context3 return nullptr");
        return false;
    }
    if (Pkt = av_packet_alloc(); Pkt == nullptr)
    {
        LOGE("av_packet_alloc return nullptr");
        return false;
    }
    if (Frame = av_frame_alloc(); Frame == nullptr)
    {
        LOGE("av_frame_alloc return nullptr");
        return false;
    }

    Frame->width  = cfg.width;
    Frame->height = cfg.height;
    Frame->format = AVPixelFormat::AV_PIX_FMT_YUV420P;
    Frame->pts    = 0;
    if (auto r = av_frame_get_buffer(Frame, 0); r != 0)
    {
        LOGE("av_frame_get_buffer return [%d]", r);
        return false;
    }
    if (auto r = av_frame_make_writable(Frame); r != 0)
    {
        LOGE("av_frame_make_writable return [%d]", r);
        return false;
    }

    CCtx->codec_id  = codec->id;
    CCtx->bit_rate  = cfg.bit_rate;
    CCtx->width     = cfg.width;
    CCtx->height    = cfg.height;
    CCtx->time_base = AVRational{1, cfg.fps};
    CCtx->pix_fmt   = AV_PIX_FMT_YUV420P;
    if (Ctx->oformat->flags & AVFMT_GLOBALHEADER) CCtx->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;

    AVDictionary *opt = nullptr;
    for (auto &o : cfg.opt)
    {
        av_dict_set(&opt, o.first.c_str(), o.second.c_str(), 0);
    }
    std::unique_ptr<AVDictionary *, decltype(av_dict_free) *> up_opt{&opt, av_dict_free};

    if (auto r = avcodec_open2(CCtx, codec, &opt); r != 0)
    {
        LOGE("avcodec_open2 return [%d]", r);
        return false;
    }
    if (auto r = avcodec_parameters_from_context(St->codecpar, CCtx); r != 0)
    {
        LOGE("avcodec_parameters_from_context return [%d]", r);
        return false;
    }

    Sws = sws_alloc_context();
    if (Sws == nullptr)
    {
        LOGE("call sws_alloc_context return nullptr");
        return false;
    }

    av_dump_format(Ctx, 0, Ctx->url, 1);

    if (auto r = avio_open(&Ctx->pb, Ctx->url, AVIO_FLAG_WRITE); r != 0)
    {
        LOGE("avio_open return [%d]", r);
        return false;
    }

    if (auto r = avformat_write_header(Ctx, &opt); r != 0)
    {
        LOGE("avformat_write_header return [%d]", r);
        return false;
    }

    NeedTailer = true;
    IsReady    = true;
    return true;
}
bool EncoderNode::FlushingEncodec()
{
    avcodec_send_frame(CCtx, NULL);
    while (true)
    {
        auto r = avcodec_receive_packet(CCtx, Pkt);
        if (r == AVERROR_EOF)
        {
            break;
        }
        else if (r == AVERROR(EAGAIN))
        {
            continue;
        }
        std::unique_ptr<AVPacket, decltype(av_packet_unref) *> unref_guard{Pkt, av_packet_unref};

        av_packet_rescale_ts(Pkt, CCtx->time_base, St->time_base);
        Pkt->stream_index = St->index;
        if (r = av_write_frame(Ctx, Pkt); r < 0)
        {
            LOGE("av_write_frame return [%d]", r);
            break;
        }
    }
    return true;
}

bool EncoderNode::PushOneFrame(const cv::Mat &image)
{
    if (not IsReady)
    {
        LOGW("not ready!");
        return false;
    }
    Sws =
        sws_getCachedContext(Sws, image.cols, image.rows, AVPixelFormat::AV_PIX_FMT_BGR24, Frame->width, Frame->height,
                             static_cast<AVPixelFormat>(Frame->format), OutFlags, nullptr, nullptr, nullptr);
    if (Sws == nullptr)
    {
        LOGW("sws is nullptr");
        return false;
    }

    int linesizes[1]{};
    linesizes[0] = image.step1();
    sws_scale(Sws, &image.data, linesizes, 0, image.rows, Frame->data, Frame->linesize);
    if (auto r = avcodec_send_frame(CCtx, Frame); r != 0)
    {
        LOGW("avcodec_send_frame return [%d]", r);
        return false;
    }
    ++Frame->pts;
    while (true)
    {
        auto r = avcodec_receive_packet(CCtx, Pkt);
        if (r == AVERROR(EAGAIN) || r == AVERROR_EOF)
        {
            break;
        }
        else if (r < 0)
        {
            LOGW("avcodec_receive_packet return [%d]", r);
            return false;
        }
        std::unique_ptr<AVPacket, decltype(av_packet_unref) *> unref_guard{Pkt, av_packet_unref};

        av_packet_rescale_ts(Pkt, CCtx->time_base, St->time_base);
        Pkt->stream_index = St->index;
        if (r = av_write_frame(Ctx, Pkt); r < 0)
        {
            LOGW("av_write_frame return [%d]", r);
            return false;
        }
    }

    return true;
}

bool EncoderNode::Close()
{
    if (CCtx != nullptr) FlushingEncodec();
    IsReady = false;

    if (Sws != nullptr)
    {
        sws_freeContext(Sws);
        Sws = nullptr;
    }
    if (Frame != nullptr)
    {
        av_frame_free(&Frame);
    }
    if (Pkt != nullptr)
    {
        av_packet_free(&Pkt);
    }
    if (NeedTailer)
    {
        av_write_trailer(Ctx);
        NeedTailer = false;
    }
    if (CCtx != nullptr)
    {
        avcodec_free_context(&CCtx);
    }
    if (Ctx != nullptr)
    {
        avio_closep(&Ctx->pb);
        avformat_free_context(Ctx);
        Ctx = nullptr;
    }
    return true;
}

bool EncoderNode::Run()
{
    while (Running)
    {
        while (Running and not IsSignalQueListReady(InputList))
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        if (not Running)
        {
            break;
        }
        auto signal = GetSignalList(InputList);
        if (signal.size() != 1)
        {
            LOGE("EncoderNode::Run() signal.size() != 1, excepted: [1], actual: [%d]", signal.size());
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }
        auto input_signal = signal.front();
        if (input_signal->GetSignalType() != SignalType::SIGNAL_IMAGE_BGR)
        {
            LOGE("EncoderNode::Run() input_signal->GetSignalType() != SignalType::SIGNAL_IMAGE_BGR");
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }
        auto image = std::dynamic_pointer_cast<SignalImageBGR>(input_signal);
        if (image == nullptr)
        {
            LOGE("EncoderNode::Run() image == nullptr");
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }
        auto frame_index = image->FrameIdx;
        CostTimer.StartTimer();
        if (not PushOneFrame(image->Val))
        {
            LOGE("EncoderNode::Run() PushOneFrame return false");
        }
        CostTimer.EndTimer();
        auto now    = std::chrono::steady_clock::now();
        auto peroid = std::chrono::duration_cast<std::chrono::milliseconds>(now - StartTime).count();
        auto fps    = 1000.0f * frame_index / peroid;
        LOGI("FrameIndex = [%d], peroid = [%ld] ms, FPS = [%f]", frame_index, peroid, fps);
    }
    return true;
};

}  // namespace cv_infer