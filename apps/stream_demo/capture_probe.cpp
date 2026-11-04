// GPU NVMM decode via DeepStream. Selected JPEG frames travel to the local gate.
// Inference is performed by the existing resident TensorRT engine, not this probe.
#include <gst/gst.h>
#include <gst/app/gstappsink.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <unistd.h>
#include <glib-unix.h>

static FILE* packets=nullptr;

struct Capture {
  GMainLoop* loop{};
  GstElement* queue{};
  GstElement* caps{};
  std::atomic<bool> failed{false};
  std::atomic<gint64> last{g_get_monotonic_time()};
  bool got_frame=false;
};
static Capture* active=nullptr;
static gboolean stop_signal(gpointer data) { g_main_loop_quit(static_cast<Capture*>(data)->loop); return G_SOURCE_REMOVE; }

static GstFlowReturn sample(GstAppSink* sink,gpointer data) {
  auto& state=*static_cast<Capture*>(data);
  GstSample* value=gst_app_sink_pull_sample(sink);
  if(!value)return GST_FLOW_EOS;
  auto* buffer=gst_sample_get_buffer(value);
  GstMapInfo map=GST_MAP_INFO_INIT;
  if(!buffer||!gst_buffer_map(buffer,&map,GST_MAP_READ)){gst_sample_unref(value);return GST_FLOW_ERROR;}
  // Dedicated binary fd: NVIDIA plugins may emit diagnostics on stdout.
  const guint64 pts=GST_BUFFER_PTS_IS_VALID(buffer)?GST_BUFFER_PTS(buffer):0;
  std::fprintf(packets,"STRW %llu %zu\n",static_cast<unsigned long long>(pts),map.size);
  const bool ok=std::fwrite(map.data,1,map.size,packets)==map.size && std::fflush(packets)==0;
  state.last=g_get_monotonic_time();state.got_frame=true;
  gst_buffer_unmap(buffer,&map);gst_sample_unref(value);
  return ok?GST_FLOW_OK:GST_FLOW_ERROR;
}
static GstPadProbeReturn negotiated_caps(GstPad*,GstPadProbeInfo* info,gpointer data){
  auto& state=*static_cast<Capture*>(data);
  auto* event=GST_PAD_PROBE_INFO_EVENT(info);
  if(event && GST_EVENT_TYPE(event)==GST_EVENT_CAPS){
    GstCaps* caps=nullptr;gst_event_parse_caps(event,&caps);
    const auto* structure=caps?gst_caps_get_structure(caps,0):nullptr;
    int width=0,height=0;
    if(structure && gst_structure_get_int(structure,"width",&width) && gst_structure_get_int(structure,"height",&height) && width>0 && height>0){
      const double scale=std::min({1.,960./width,720./height});
      width=std::max(2,static_cast<int>(width*scale)/2*2);height=std::max(2,static_cast<int>(height*scale)/2*2);
      GstCaps* target=gst_caps_new_simple("video/x-raw","format",G_TYPE_STRING,"I420","width",G_TYPE_INT,width,
        "height",G_TYPE_INT,height,"pixel-aspect-ratio",GST_TYPE_FRACTION,1,1,nullptr);
      g_object_set(state.caps,"caps",target,nullptr);gst_caps_unref(target);
    }
  }
  return GST_PAD_PROBE_OK;
}
static void source_pad(GstElement*,GstPad* pad,gpointer data) {
  auto& state=*static_cast<Capture*>(data);
  GstCaps* caps=gst_pad_get_current_caps(pad);
  if(!caps)caps=gst_pad_query_caps(pad,nullptr);
  const auto* structure=caps?gst_caps_get_structure(caps,0):nullptr;
  if(!structure||!g_str_has_prefix(gst_structure_get_name(structure),"video/")){if(caps)gst_caps_unref(caps);return;}
  if(!gst_caps_features_contains(gst_caps_get_features(caps,0),"memory:NVMM")){
    std::cerr<<"Source did not provide GPU NVMM; no CPU fallback\n";
    state.failed=true;g_main_loop_quit(state.loop);gst_caps_unref(caps);return;
  }
  gst_caps_unref(caps);
  gst_pad_add_probe(pad,GST_PAD_PROBE_TYPE_EVENT_DOWNSTREAM,negotiated_caps,&state,nullptr);
  auto* sink=gst_element_get_static_pad(state.queue,"sink");
  if(gst_pad_link(pad,sink)!=GST_PAD_LINK_OK){state.failed=true;g_main_loop_quit(state.loop);}
  gst_object_unref(sink);
}
static gboolean bus(GstBus*,GstMessage* message,gpointer data){
  auto& state=*static_cast<Capture*>(data);
  if(GST_MESSAGE_TYPE(message)==GST_MESSAGE_ERROR){
    // Source URIs may contain credentials. Do not emit the GStreamer debug text.
    std::cerr<<"DeepStream capture failed; check source/codec/network\n";
    state.failed=true;g_main_loop_quit(state.loop);
  }else if(GST_MESSAGE_TYPE(message)==GST_MESSAGE_EOS)g_main_loop_quit(state.loop);
  return G_SOURCE_CONTINUE;
}
static gboolean watchdog(gpointer data){
  auto& state=*static_cast<Capture*>(data);
  if(g_get_monotonic_time()-state.last.load()>20*G_USEC_PER_SEC){
    state.failed=true;std::cerr<<"No decoded video frame for 20 seconds\n";g_main_loop_quit(state.loop);
  }
  return G_SOURCE_CONTINUE;
}
int main(int argc,char** argv){
  const int output=dup(STDOUT_FILENO);dup2(STDERR_FILENO,STDOUT_FILENO);packets=fdopen(output,"wb");
  gst_init(&argc,&argv);
  const char* uri=std::getenv("STRAW_CAPTURE_URI");if(!uri||!*uri)return 2;
  Capture state;active=&state;state.loop=g_main_loop_new(nullptr,FALSE);
  g_unix_signal_add(SIGTERM,stop_signal,&state);g_unix_signal_add(SIGINT,stop_signal,&state);std::signal(SIGPIPE,SIG_IGN);
  auto* pipeline=gst_pipeline_new("straw-gpu-capture");
  auto* source=gst_element_factory_make("nvurisrcbin","source");
  state.queue=gst_element_factory_make("queue","capture_queue");
  auto* convert=gst_element_factory_make("nvvideoconvert","convert");
  state.caps=gst_element_factory_make("capsfilter","shape");
  auto* rate=gst_element_factory_make("videorate","rate");
  auto* fps=gst_element_factory_make("capsfilter","fps");
  auto* jpeg=gst_element_factory_make("jpegenc","jpeg");
  auto* sink=gst_element_factory_make("appsink","sink");
  if(!pipeline||!source||!state.queue||!convert||!state.caps||!rate||!fps||!jpeg||!sink)return 2;
  g_object_set(source,"uri",uri,"gpu-id",0,"cudadec-memtype",0,nullptr);
  if(g_str_has_prefix(uri,"rtsp://")||g_str_has_prefix(uri,"rtsps://"))
    g_object_set(source,"select-rtp-protocol",4,"latency",200,"rtsp-reconnect-interval",5,nullptr);
  g_object_set(state.queue,"max-size-buffers",2,"max-size-bytes",0,"max-size-time",static_cast<guint64>(0),nullptr);
  g_object_set(rate,"drop-only",TRUE,nullptr);
  auto* initialcaps=gst_caps_from_string("video/x-raw,format=I420");g_object_set(state.caps,"caps",initialcaps,nullptr);gst_caps_unref(initialcaps);
  auto* fpscaps=gst_caps_from_string("video/x-raw,framerate=30/1");g_object_set(fps,"caps",fpscaps,nullptr);gst_caps_unref(fpscaps);
  g_object_set(jpeg,"quality",90,nullptr);
  g_object_set(sink,"sync",TRUE,"async",FALSE,"max-buffers",2,"drop",TRUE,"wait-on-eos",FALSE,nullptr);
  GstAppSinkCallbacks callbacks{};callbacks.new_sample=sample;gst_app_sink_set_callbacks(GST_APP_SINK(sink),&callbacks,&state,nullptr);
  gst_bin_add_many(GST_BIN(pipeline),source,state.queue,convert,state.caps,rate,fps,jpeg,sink,nullptr);
  if(!gst_element_link_many(state.queue,convert,state.caps,rate,fps,jpeg,sink,nullptr))return 2;
  g_signal_connect(source,"pad-added",G_CALLBACK(source_pad),&state);
  auto* pipelinebus=gst_element_get_bus(pipeline);gst_bus_add_watch(pipelinebus,bus,&state);gst_object_unref(pipelinebus);
  g_timeout_add_seconds(1,watchdog,&state);
  if(gst_element_set_state(pipeline,GST_STATE_PLAYING)==GST_STATE_CHANGE_FAILURE)state.failed=true;
  else g_main_loop_run(state.loop);
  gst_element_set_state(pipeline,GST_STATE_NULL);gst_object_unref(pipeline);g_main_loop_unref(state.loop);
  return state.failed?3:0;
}
