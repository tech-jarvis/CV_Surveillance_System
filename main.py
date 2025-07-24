import asyncio
import base64
import json
import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor
import uuid
from queue import Empty

import cv2
from fastapi import BackgroundTasks, FastAPI, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from websockets.exceptions import ConnectionClosedOK

from analytics_channel import channel_analytics, live_channel_heatmap
from analytics_video import analytics, live_heatmap
from config import VIDEO_PATH
from stats import historic_stats
from utils import image_to_base64

app = FastAPI(debug=True)

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pool = ProcessPoolExecutor()

import zmq
import zmq.asyncio

async def zmq_subscriber(websocket: WebSocket, PORT):
    context = zmq.asyncio.Context()
    socket = context.socket(zmq.SUB)
    socket.connect(f"tcp://localhost:{PORT}")  # Change to your publisher address
    socket.setsockopt_string(zmq.SUBSCRIBE, "")

    try:
        while True:
            message = await socket.recv_string()
            await websocket.send_text(message)
    except Exception as e:
        print(f"Error: {e}")
    finally:
        socket.close()
        context.term()

@app.websocket("/video")
async def websocket_endpoint(
    websocket: WebSocket,
    filename: str = Query(...),
    start_frame: int = Query(0),
    skip_frame: int = Query(1),
    fps: int = Query(1),
    show_frame: bool = Query(False),
    cashier_recognition: bool = Query(False),
    display_region: bool = Query(True),
):
    try:
        await websocket.accept()
        print(filename, skip_frame, fps, start_frame)

        async for output in analytics(
            video_name=filename,
            start_frame=start_frame,
            skip_frame=skip_frame,
            show_frame=show_frame,
            cashier_recognition=cashier_recognition,
            display_region=display_region,
        ):
            frame = output["frame"]

            if show_frame:
                frame = image_to_base64(frame)

            await websocket.send_json({"frame": frame, "metadata": output["metadata"]})
            await asyncio.sleep(1 / fps)

    except WebSocketDisconnect:
        print("WebSocket disconnected")
    except RuntimeError as e:
        print("Runtime Error", e)
    except ConnectionClosedOK:
        pass
    # except Exception as e:
    #     print("Error", e, filename, skip_frame, fps, show_frame)


async def bg_channel(
    que: mp.Queue,
    channel_name,
    start_frame,
    skip_frame,
    show_frame,
    cashier_recognition,
    display_region,
) -> dict:
    for output in channel_analytics(
        channel_name=channel_name,
        start_frame=start_frame,
        skip_frame=skip_frame,
        show_frame=show_frame,
        cashier_recognition=cashier_recognition,
        display_region=display_region,
    ):
        print(2)
        frame = output["frame"]

        if show_frame:
            frame = image_to_base64(frame)

        que.put(json.dumps({"frame": frame,"metadata": output["metadata"]}))

        if que.qsize() > 300:
            break

        time.sleep(1/100)


    del playing_videos[playing_videos.index(channel_name)]

m_manager = mp.Manager()
playing_videos = m_manager.list()
channel_queue = m_manager.dict()


ZMQ_PUBLIHSERS = {
    "ch03": 5001,
    "ch16": 5002
}


@app.websocket("/channel")
async def websocket_channel_endpoint(
    websocket: WebSocket,
    channel_name: str = Query(...),
    start_frame: int = Query(0),
    skip_frame: int = Query(1),
    fps: int = Query(1),
    show_frame: bool = Query(False),
    cashier_recognition: bool = Query(False),
    display_region: bool = Query(True),
):

    try:
        await websocket.accept()
        await zmq_subscriber(websocket, ZMQ_PUBLIHSERS[channel_name])
    except WebSocketDisconnect:
        print("Websocket Disconnected!")
    except Exception as ex:
        print(ex)
        
# @app.websocket("/channel")
# async def websocket_channel_endpoint(
#     websocket: WebSocket,
#     channel_name: str = Query(...),
#     start_frame: int = Query(0),
#     skip_frame: int = Query(1),
#     fps: int = Query(1),
#     show_frame: bool = Query(False),
#     cashier_recognition: bool = Query(False),
#     display_region: bool = Query(True),
# ):
#     # loop = asyncio.get_event_loop()

#     m_manager = mp.Manager()
#     que = m_manager.Queue()

#     start_time = time.time()
#     try:
#         await websocket.accept()
#         print("Channel: ", channel_name, skip_frame, fps, start_frame)

#         # result = loop.run_in_executor(
#         #     pool,
#         #     bg_channel,
#         #     que,
#         #     channel_name,
#         #     start_frame,
#         #     skip_frame,
#         #     show_frame,
#         #     cashier_recognition,
#         #     display_region,
#         # )

#         await bg_channel

#         while True:
#             try:
#                 q_result = que.get(block=False)
#                 try:
#                     await websocket.send_json(json.loads(q_result))
#                     # print(websocket.client, time.time() - start_time)
#                 except WebSocketDisconnect:
#                     print("WebSocket disconnected")
#                     break
#             except Empty:
#                 q_result = None
#                 pass
#             # await websocket.send_json(json.loads(result))
#             if result.done():
#                 pass

#             await asyncio.sleep(0)

#             # print(websocket.client, time.time()-start_time)
#             start_time = time.time()

#     except WebSocketDisconnect:
#         print("WebSocket disconnected")
#     except RuntimeError as e:
#         print("Runtime Error", e)
#     except ConnectionClosedOK:
#         pass
#     # except Exception as e:
#     #     print("Error", e, filename, skip_frame, fps, show_frame)


@app.websocket("/channel_old")
async def websocket_channel_endpoint_old(
    websocket: WebSocket,
    channel_name: str = Query(...),
    start_frame: int = Query(0),
    skip_frame: int = Query(1),
    fps: int = Query(1),
    show_frame: bool = Query(False),
    cashier_recognition: bool = Query(False),
    display_region: bool = Query(True),
):
    start_time = time.time()
    try:
        await websocket.accept()
        print("Channel: ", channel_name, skip_frame, fps, start_frame)

        async for output in channel_analytics(
            channel_name=channel_name,
            start_frame=start_frame,
            skip_frame=skip_frame,
            show_frame=show_frame,
            cashier_recognition=cashier_recognition,
            display_region=display_region,
        ):
            frame = output["frame"]

            if show_frame:
                frame = image_to_base64(frame)

            await websocket.send_json({"frame": frame, "metadata": output["metadata"]})
            await asyncio.sleep(0)

            # print(websocket.client, time.time() - start_time)
            # start_time = time.time()

    except WebSocketDisconnect:
        print("WebSocket disconnected")
    except RuntimeError as e:
        print("Runtime Error", e)
    except ConnectionClosedOK:
        pass
    # except Exception as e:
    #     print("Error", e, filename, skip_frame, fps, show_frame)


@app.websocket("/live_channel_heatmap")
async def live_channel_heatmap_endpoint(
    websocket: WebSocket,
    channel_name: str = Query(...),
    heatmap_delay: int = Query(10),
    fps: int = Query(1),
    target_label: str = Query(None),
):
    try:
        await websocket.accept()

        async for heatmap in live_channel_heatmap(
            channel_name=channel_name,
            heatmap_delay=heatmap_delay,
            target_label=target_label,
        ):
            heatmap = image_to_base64(heatmap)

            await websocket.send_json({"heatmap": heatmap})
            await asyncio.sleep(0)

    except WebSocketDisconnect:
        print("WebSocket disconnected")
    except RuntimeError as e:
        print("Runtime Error", e)
    except ConnectionClosedOK:
        pass
    except Exception as e:
        print("Error", e, channel_name, fps, heatmap_delay)


@app.get("/stats")
async def get_stats(video_name: str = Query(...)):
    return JSONResponse(historic_stats(video_name))


@app.websocket("/live_heatmap")
async def live_heatmap_endpoint(
    websocket: WebSocket,
    filename: str = Query(...),
    heatmap_delay: int = Query(10),
    fps: int = Query(1),
    target_label: str = Query(None),
):
    try:
        await websocket.accept()

        async for heatmap in live_heatmap(
            video_name=filename, heatmap_delay=heatmap_delay, target_label=target_label
        ):
            heatmap = image_to_base64(heatmap)

            await websocket.send_json({"heatmap": heatmap})
            await asyncio.sleep(1 / fps)

    except WebSocketDisconnect:
        print("WebSocket disconnected")
    except RuntimeError as e:
        print("Runtime Error", e)
    except ConnectionClosedOK:
        pass
    except Exception as e:
        print("Error", e, filename, fps, heatmap_delay)


async def generate_heatmap_stream(video_name: str):
    for indx, heatmap in enumerate(live_heatmap(video_name=video_name)):
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + cv2.imencode(".jpg", heatmap, [cv2.IMWRITE_JPEG_QUALITY, 30])[1].tobytes()
            + b"\r\n"
        )
        await asyncio.sleep(1 / 1)


@app.get("/live_heatmap")
async def get_live_heatmap(video_name: str = Query(...)):
    return StreamingResponse(
        generate_heatmap_stream(video_name),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


async def generate_image_stream(channel_name: str, skip_frame: int):
    async for output in channel_analytics(
        channel_name=channel_name,
        start_frame=0,
        skip_frame=skip_frame,
        cashier_recognition=True,
        show_frame=True,
    ):
        frame = output["frame"]
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 30])[1].tobytes()
            + b"\r\n"
        )

        await asyncio.sleep(1 / 25)


@app.get("/broadcast")
async def get_broadcast_video(
    channel_name: str = Query(...), skip_frame: int = Query(1)
):
    return StreamingResponse(
        generate_image_stream(channel_name, skip_frame),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/videos_list")
async def get_videos_list():
    return JSONResponse({"videos_list": os.listdir(VIDEO_PATH)})


@app.websocket("/live_video")
async def live_video(
    websocket: WebSocket,
    filename: str = Query(...),
    skip_frame: int = Query(25),
    fps: int = Query(1),
):
    try:
        await websocket.accept()
        while True:
            indx = 0
            cap = cv2.VideoCapture(f"{VIDEO_PATH}/{filename}")

            while True:
                indx += 1

                if indx % skip_frame != 0:
                    continue

                ret, frame = cap.read()

                height, width, _ = frame.shape
                target_width = 320
                ratio = target_width / width

                frame = cv2.resize(frame, dsize=None, fx=ratio, fy=ratio)

                ret, buffer = cv2.imencode(
                    ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 50]
                )

                frame_base64 = base64.b64encode(buffer).decode("utf-8")
                frame_data = f"data:image/jpeg;base64,{frame_base64}"

                await websocket.send_json({"frame": frame_data})
                await asyncio.sleep(0)

    except WebSocketDisconnect:
        print("WebSocket disconnected")
    except Exception as e:
        print("Error", e)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8009,
        # log_level="debug",
        ws_ping_timeout=None
        )
