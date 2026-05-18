import 'dart:async';
import 'package:camera/camera.dart';
import 'package:flutter/foundation.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:image/image.dart' as img;

class ModelService {
  Interpreter? _interpreter;
  bool _isModelLoaded = false;
  List<int> _inputShape = [1, 256, 256, 3];
  List<int> _outputShape = [1, 256, 256, 3];

  bool get isModelLoaded => _isModelLoaded;

  /// Direct Buffer Loading (The most stable method on Poco X3 Pro).
  /// We bypass standard asset loading to avoid platform binding races.
  Future<void> loadModelFromBuffer(Uint8List modelBuffer) async {
    if (_isModelLoaded) return;
    
    // Robust 1s delay for Poco/Xiaomi native libs
    await Future.delayed(const Duration(milliseconds: 1000));

    try {
      // Diagnostic check: Ensure correct TFLite signature (TFL3)
      if (modelBuffer.length < 8) throw "Model buffer is too small";
      final sig = String.fromCharCodes(modelBuffer.sublist(4, 8));
      if (sig != "TFL3") throw "Model has invalid signature: $sig";

      _interpreter = Interpreter.fromBuffer(
        modelBuffer,
        options: InterpreterOptions(),
      );
      
      _isModelLoaded = true;
      _extractTensorInfo();
      debugPrint("✅ Model Service: Initialized via Direct Buffer (v3)");
    } catch (e) {
      debugPrint("❌ Model Service: Initialization Failure: $e");
      rethrow;
    }
  }

  void _extractTensorInfo() {
    final inputTensors = _interpreter!.getInputTensors();
    final outputTensors = _interpreter!.getOutputTensors();
    if (inputTensors.isNotEmpty) {
      _inputShape = List<int>.from(inputTensors[0].shape);
      if (_inputShape[0] <= 0) _inputShape[0] = 1;
    }
    if (outputTensors.isNotEmpty) {
      _outputShape = List<int>.from(outputTensors[0].shape);
      if (_outputShape[0] <= 0) _outputShape[0] = 1;
    }
  }

  Uint8List? processCameraImage(CameraImage image) {
    if (!_isModelLoaded || _interpreter == null) return null;

    try {
      final input = _convertRawToFloat32(image);
      return _runInferenceOnFloat32(input);
    } catch (e) {
      debugPrint("❌ Model Service: Runtime Failure: $e");
      return null;
    }
  }

  Uint8List? _runInferenceOnFloat32(Float32List input) {
    final bool isNHWC = _inputShape.length == 4 && _inputShape[3] == 3;
    final int batch = _inputShape[0];
    final int h = isNHWC ? _inputShape[1] : _inputShape[2];
    final int w = isNHWC ? _inputShape[2] : _inputShape[3];
    final int c = 3;

    Object finalInput;
    if (isNHWC) {
      final nhwcInput = Float32List(batch * h * w * c);
      for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
          final int srcIdx = y * w + x;
          final int dstIdx = (y * w + x) * 3;
          if (srcIdx + (2 * h * w) < input.length) {
            nhwcInput[dstIdx] = input[srcIdx]; 
            nhwcInput[dstIdx + 1] = input[srcIdx + (h * w)]; 
            nhwcInput[dstIdx + 2] = input[srcIdx + (2 * h * w)];
          }
        }
      }
      finalInput = nhwcInput.reshape(_inputShape);
    } else {
      finalInput = input.reshape(_inputShape);
    }

    final outputSize = _outputShape.reduce((a, b) => a * b);
    final outputBuffer = Float32List(outputSize).reshape(_outputShape);
    
    _interpreter!.run(finalInput, outputBuffer);

    final resultImage = img.Image(width: 256, height: 256);
    final bool outNHWC = _outputShape.length == 4 && _outputShape[3] == 3;
    
    for (int y = 0; y < 256; y++) {
      for (int x = 0; x < 256; x++) {
        double r, g, b;
        if (outNHWC) {
          r = (outputBuffer[0][y][x][0] as num).toDouble();
          g = (outputBuffer[0][y][x][1] as num).toDouble();
          b = (outputBuffer[0][y][x][2] as num).toDouble();
        } else {
          r = (outputBuffer[0][0][y][x] as num).toDouble();
          g = (outputBuffer[0][1][y][x] as num).toDouble();
          b = (outputBuffer[0][2][y][x] as num).toDouble();
        }
        resultImage.setPixel(x, y, img.ColorRgb8(
          ((r + 1.0) * 127.5).clamp(0, 255).toInt(), 
          ((g + 1.0) * 127.5).clamp(0, 255).toInt(), 
          ((b + 1.0) * 127.5).clamp(0, 255).toInt()
        ));
      }
    }
    return Uint8List.fromList(img.encodeJpg(resultImage, quality: 75));
  }

  Float32List _convertRawToFloat32(CameraImage image) {
    final input = Float32List(1 * 3 * 256 * 256);
    final width = image.width;
    final height = image.height;
    final double xStep = width / 256;
    final double yStep = height / 256;

    for (int y = 0; y < 256; y++) {
      for (int x = 0; x < 256; x++) {
        final int srcX = (x * xStep).toInt();
        final int srcY = (y * yStep).toInt();
        
        double r, g, b;
        if (image.format.group == ImageFormatGroup.yuv420) {
          final planeY = image.planes[0];
          final planeU = image.planes[1];
          final planeV = image.planes[2];
          
          final int yIndex = srcY * planeY.bytesPerRow + srcX * (planeY.bytesPerPixel ?? 1);
          final int uvX = srcX ~/ 2;
          final int uvY = srcY ~/ 2;
          final int uvIndex = uvY * planeU.bytesPerRow + uvX * (planeU.bytesPerPixel ?? 1);
          
          if (yIndex < planeY.bytes.length && uvIndex < planeU.bytes.length && uvIndex < planeV.bytes.length) {
            final int yValue = planeY.bytes[yIndex];
            final int uValue = planeU.bytes[uvIndex];
            final int vValue = planeV.bytes[uvIndex];
            r = (yValue + 1.13983 * (vValue - 128)).clamp(0, 255);
            g = (yValue - 0.39465 * (uValue - 128) - 0.58060 * (vValue - 128)).clamp(0, 255);
            b = (yValue + 2.03211 * (uValue - 128)).clamp(0, 255);
          } else {
            r = g = b = 0;
          }
        } else {
          final plane = image.planes[0];
          final int pixelIdx = srcY * plane.bytesPerRow + srcX * (plane.bytesPerPixel ?? 4);
          if (pixelIdx + 2 < plane.bytes.length) {
            b = plane.bytes[pixelIdx].toDouble();
            g = plane.bytes[pixelIdx + 1].toDouble();
            r = plane.bytes[pixelIdx + 2].toDouble();
          } else {
            r = g = b = 0;
          }
        }
        
        final int inputIdx = y * 256 + x;
        input[inputIdx] = (r / 127.5) - 1.0;
        input[inputIdx + (256 * 256)] = (g / 127.5) - 1.0;
        input[inputIdx + (2 * 256 * 256)] = (b / 127.5) - 1.0;
      }
    }
    return input;
  }

  void dispose() {
    _interpreter?.close();
  }
}
