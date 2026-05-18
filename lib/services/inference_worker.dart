import 'dart:async';
import 'dart:isolate';
import 'dart:io';
import 'package:flutter/foundation.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:image/image.dart' as img;

/// Data packet for camera frames
class InferenceFrame {
  final Uint8List y;
  final Uint8List u;
  final Uint8List v;
  final int width;
  final int height;
  final int yRowStride;
  final int uvRowStride;
  final int uvPixelStride;
  final String format;
  final int rotation;

  InferenceFrame({
    required this.y,
    required this.u,
    required this.v,
    required this.width,
    required this.height,
    required this.yRowStride,
    required this.uvRowStride,
    required this.uvPixelStride,
    required this.format,
    this.rotation = 0,
  });
}

class InferenceWorker {
  Isolate? _isolate;
  SendPort? _sendPort;
  final _receivePort = ReceivePort();
  
  bool _isReady = false;
  bool _isProcessing = false;

  final _resultController = StreamController<dynamic>.broadcast();
  Stream<dynamic> get results => _resultController.stream;

  Future<void> start(Uint8List modelBytes) async {
    _isolate = await Isolate.spawn(_isolateEntryPoint, {
      'sendPort': _receivePort.sendPort,
      'modelBytes': modelBytes,
    });

    _receivePort.listen((message) {
      if (message is SendPort) {
        _sendPort = message;
        _isReady = true;
      } else if (message is Uint8List) {
        _resultController.add(message);
        _isProcessing = false;
      } else if (message is String && message.startsWith("ERROR")) {
        debugPrint("❌ Isolate Error: $message");
        _isProcessing = false;
      }
    });
  }

  void process(InferenceFrame frame) {
    if (!_isReady || _isProcessing || _sendPort == null) return;
    _isProcessing = true;
    _sendPort!.send(frame);
  }

  static void _isolateEntryPoint(Map<String, dynamic> initData) {
    final SendPort mainSendPort = initData['sendPort'];
    final Uint8List modelBytes = initData['modelBytes'];

    final isolateReceivePort = ReceivePort();
    mainSendPort.send(isolateReceivePort.sendPort);

    Interpreter? interpreter;
    
    try {
      final options = InterpreterOptions()..threads = 4;
      
      // 🚀 GPU Acceleration for iPhone only (Metal / Neural Engine)
      // Note: tflite_flutter GPU bindings only support Android/iOS, not macOS
      if (Platform.isIOS) {
        options.addDelegate(GpuDelegateV2());
      }
      
      interpreter = Interpreter.fromBuffer(modelBytes, options: options);
      mainSendPort.send("READY");
    } catch (e) {
      mainSendPort.send("ERROR: Model load failed: $e");
      return;
    }

    final inputTensors = interpreter.getInputTensors();
    final outputTensors = interpreter.getOutputTensors();
    final inputShape = inputTensors[0].shape;
    final outputShape = outputTensors[0].shape;
    
    // Detect if model is NHWC or NCHW
    final bool isInputNHWC = inputShape.length == 4 && inputShape[3] == 3;
    final bool isOutputNHWC = outputShape.length == 4 && outputShape[3] == 3;

    isolateReceivePort.listen((message) {
      if (message is InferenceFrame) {
        try {
          final Float32List inputBuffer = _convertFrameToFloat32(message, isInputNHWC);
          final input = inputBuffer.reshape(inputShape);
          
          final outputSize = outputShape.reduce((a, b) => a * b);
          final outputBuffer = Float32List(outputSize).reshape(outputShape);
          
          interpreter!.run(input, outputBuffer);

          final result = _convertOutputToJpg(outputBuffer, isOutputNHWC);
          mainSendPort.send(result);
        } catch (e) {
          mainSendPort.send("ERROR: Inference failed: $e");
        }
      }
    });
  }

  static Float32List _convertFrameToFloat32(InferenceFrame frame, bool toNHWC) {
    final int h = 256;
    final int w = 256;
    final input = Float32List(1 * 3 * h * w);
    
    final double xStep = frame.width / w;
    final double yStep = frame.height / h;

    for (int y = 0; y < h; y++) {
      for (int x = 0; x < w; x++) {
        // Apply rotation if needed
        int rotX = x;
        int rotY = y;

        if (frame.rotation == 90) {
          rotX = y;
          rotY = w - 1 - x;
        } else if (frame.rotation == 180) {
          rotX = w - 1 - x;
          rotY = h - 1 - y;
        } else if (frame.rotation == 270) {
          rotX = h - 1 - y;
          rotY = x;
        }

        final int srcX = (rotX * xStep).toInt();
        final int srcY = (rotY * yStep).toInt();
        
        double r, g, b;
        if (frame.format == 'yuv420') {
          final int yIdx = (srcY * frame.yRowStride + srcX).clamp(0, frame.y.length - 1);
          final int uvX = srcX ~/ 2;
          final int uvY = srcY ~/ 2;
          final int uvIdx = (uvY * frame.uvRowStride + uvX * frame.uvPixelStride).clamp(0, frame.u.length - 1);
          
          final int yVal = frame.y[yIdx];
          final int uVal = frame.u[uvIdx];
          final int vVal = frame.v[uvIdx];
          
          r = (yVal + 1.402 * (vVal - 128)).clamp(0, 255);
          g = (yVal - 0.344136 * (uVal - 128) - 0.714136 * (vVal - 128)).clamp(0, 255);
          b = (yVal + 1.772 * (uVal - 128)).clamp(0, 255);
        } else {
          final int pixelIdx = (srcY * frame.yRowStride + srcX * 4).clamp(0, frame.y.length - 4);
          b = frame.y[pixelIdx].toDouble();
          g = frame.y[pixelIdx + 1].toDouble();
          r = frame.y[pixelIdx + 2].toDouble();
        }
        
        final double nr = (r / 127.5) - 1.0;
        final double ng = (g / 127.5) - 1.0;
        final double nb = (b / 127.5) - 1.0;

        if (toNHWC) {
          final int idx = (y * w + x) * 3;
          input[idx] = nr;
          input[idx + 1] = ng;
          input[idx + 2] = nb;
        } else {
          final int base = y * w + x;
          input[base] = nr;
          input[base + (h * w)] = ng;
          input[base + (2 * h * w)] = nb;
        }
      }
    }
    return input;
  }

  static Uint8List _convertOutputToJpg(List<dynamic> output, bool isNHWC) {
    final image = img.Image(width: 256, height: 256);
    
    for (int y = 0; y < 256; y++) {
      for (int x = 0; x < 256; x++) {
        double r, g, b;
        if (isNHWC) {
          r = (output[0][y][x][0] as num).toDouble();
          g = (output[0][y][x][1] as num).toDouble();
          b = (output[0][y][x][2] as num).toDouble();
        } else {
          r = (output[0][0][y][x] as num).toDouble();
          g = (output[0][1][y][x] as num).toDouble();
          b = (output[0][2][y][x] as num).toDouble();
        }

        // De-normalization [0, 255]
        image.setPixel(x, y, img.ColorRgb8(
          ((r + 1.0) * 127.5).clamp(0, 255).toInt(), 
          ((g + 1.0) * 127.5).clamp(0, 255).toInt(), 
          ((b + 1.0) * 127.5).clamp(0, 255).toInt(),
        ));
      }
    }
    return Uint8List.fromList(img.encodeJpg(image, quality: 80));
  }

  void stop() {
    _isolate?.kill(priority: Isolate.immediate);
    _resultController.close();
  }
}
