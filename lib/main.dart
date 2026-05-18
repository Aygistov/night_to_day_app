import 'dart:io';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:night_to_day/services/inference_worker.dart';

// Conditional imports for platform-specific camera
import 'package:camera/camera.dart' as mobile_camera;
import 'package:camera_macos/camera_macos.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();

  // Camera list is only needed for mobile (iOS/Android)
  List<mobile_camera.CameraDescription> cameras = [];
  if (!Platform.isMacOS) {
    cameras = await mobile_camera.availableCameras();
  }

  runApp(NightAIApp(cameras: cameras));
}

class NightAIApp extends StatelessWidget {
  final List<mobile_camera.CameraDescription> cameras;
  const NightAIApp({super.key, required this.cameras});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Night AI',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        brightness: Brightness.dark,
        primaryColor: Colors.black,
        scaffoldBackgroundColor: Colors.black,
      ),
      home: CameraScreen(cameras: cameras),
    );
  }
}

enum AppMode {
  neutral,
  nightToDay,
}

class CameraScreen extends StatefulWidget {
  final List<mobile_camera.CameraDescription> cameras;
  const CameraScreen({super.key, required this.cameras});

  @override
  State<CameraScreen> createState() => _CameraScreenState();
}

class _CameraScreenState extends State<CameraScreen> {
  // Mobile camera (iOS/Android)
  mobile_camera.CameraController? _controller;
  
  // macOS camera
  CameraMacOSController? _macController;
  final GlobalKey _macCameraKey = GlobalKey();

  final InferenceWorker _worker = InferenceWorker();
  
  bool _isCameraInitialized = false;
  AppMode _currentMode = AppMode.neutral;
  Uint8List? _realtimeImage;
  bool _isStreaming = false;
  String? _lastError;
  int _fpsCount = 0;
  DateTime _lastFpsTime = DateTime.now();
  double _currentFps = 0;
  bool _isProcessing = false;

  @override
  void initState() {
    super.initState();
    _initInference();
    
    _worker.results.listen((dynamic message) {
      if (mounted) {
        if (message is Uint8List) {
          setState(() {
            _realtimeImage = message;
            _isProcessing = false;
            _fpsCount++;
            if (DateTime.now().difference(_lastFpsTime).inSeconds >= 1) {
              _currentFps = _fpsCount.toDouble();
              _fpsCount = 0;
              _lastFpsTime = DateTime.now();
            }
          });
        } else if (message is String) {
          if (message.startsWith("ERROR")) {
            setState(() => _lastError = message);
          } else if (message == "READY") {
            setState(() => _lastError = "AI Ready. Toggle Mode.");
          }
        }
      }
    }, onError: (err) {
      if (mounted) setState(() => _lastError = "Worker Error: $err");
    });

    // Initialize camera based on platform
    if (Platform.isMacOS) {
      // macOS camera is initialized via CameraMacOSView widget
      setState(() => _isCameraInitialized = true);
    } else {
      _initializeMobileCamera();
    }
  }

  Future<void> _initInference() async {
    try {
      setState(() => _lastError = "Loading AI Isolate...");
      final modelData = await rootBundle.load('assets/models/night_vision_optimized.tflite');
      final modelBytes = Uint8List.fromList(modelData.buffer.asUint8List(modelData.offsetInBytes, modelData.lengthInBytes));
      await _worker.start(modelBytes);
      if (mounted) setState(() => _lastError = "AI Engine Initialized. Select Mode.");
    } catch (e) {
      if (mounted) setState(() => _lastError = "Critical Init error: $e");
    }
  }

  Future<void> _initializeMobileCamera() async {
    if (widget.cameras.isEmpty) return;
    
    final imageFormat = Platform.isAndroid 
        ? mobile_camera.ImageFormatGroup.yuv420 
        : mobile_camera.ImageFormatGroup.bgra8888;

    final controller = mobile_camera.CameraController(
      widget.cameras[0], 
      mobile_camera.ResolutionPreset.low, 
      enableAudio: false,
      imageFormatGroup: imageFormat,
    );
    
    try {
      await controller.initialize();
      if (mounted) {
        setState(() {
          _controller = controller;
          _isCameraInitialized = true;
        });
      }
    } catch (e) {
      debugPrint("Camera error: $e");
    }
  }

  void _onModeChanged(AppMode mode) async {
    if (mode == _currentMode) return;
    
    if (mode == AppMode.neutral) {
      await _stopStream();
    } else {
      await _startStream();
    }

    setState(() {
      _currentMode = mode;
      _realtimeImage = null;
    });
  }

  // ========== macOS Image Stream ==========
  Future<void> _startMacStream() async {
    if (_macController == null || _isStreaming) return;
    try {
      _macController!.startImageStream((CameraImageData? imageData) {
        if (imageData == null) return;
        if (_currentMode != AppMode.neutral && !_isProcessing) {
          _isProcessing = true;

          // camera_macos streams ARGB8888 data
          // We pack it as 'bgra8888' and send plane[0] = all ARGB bytes
          final Uint8List bytes = Uint8List.fromList(imageData.bytes);
          
          _worker.process(InferenceFrame(
            y: bytes,
            u: Uint8List(0),
            v: Uint8List(0),
            width: imageData.width,
            height: imageData.height,
            yRowStride: imageData.bytesPerRow,
            uvRowStride: 0,
            uvPixelStride: 1,
            format: 'bgra8888',
            rotation: 0,
          ));
        }
      });
      setState(() => _isStreaming = true);
    } catch (e) {
      debugPrint("macOS stream error: $e");
      setState(() => _lastError = "macOS stream error: $e");
    }
  }

  Future<void> _stopMacStream() async {
    if (_macController == null || !_isStreaming) return;
    try {
      _macController!.stopImageStream();
      if (mounted) {
        setState(() {
          _isStreaming = false;
          _realtimeImage = null;
        });
      }
    } catch (e) {
      debugPrint("macOS stop error: $e");
    }
  }

  // ========== Mobile Image Stream ==========
  Future<void> _startMobileStream() async {
    if (_controller == null || _isStreaming) return;
    try {
      await _controller!.startImageStream((image) {
        if (_currentMode != AppMode.neutral && !_isProcessing) {
          _isProcessing = true;
          
          final y = Uint8List.fromList(image.planes[0].bytes);
          final u = image.planes.length > 1 ? Uint8List.fromList(image.planes[1].bytes) : Uint8List(0);
          final v = image.planes.length > 2 ? Uint8List.fromList(image.planes[2].bytes) : Uint8List(0);

          _worker.process(InferenceFrame(
            y: y,
            u: u,
            v: v,
            width: image.width,
            height: image.height,
            yRowStride: image.planes[0].bytesPerRow,
            uvRowStride: image.planes.length > 1 ? image.planes[1].bytesPerRow : 0,
            uvPixelStride: image.planes.length > 1 ? (image.planes[1].bytesPerPixel ?? 1) : 1,
            format: Platform.isAndroid ? 'yuv420' : 'bgra8888',
            rotation: 90,
          ));
        }
      });
      setState(() => _isStreaming = true);
    } catch (e) {
      debugPrint("Stream error: $e");
    }
  }

  Future<void> _stopMobileStream() async {
    if (_controller == null || !_isStreaming) return;
    try {
      await _controller!.stopImageStream();
      if (mounted) {
        setState(() {
          _isStreaming = false;
          _realtimeImage = null;
        });
      }
    } catch (e) {
      debugPrint("Stop error: $e");
    }
  }

  // ========== Unified Start/Stop ==========
  Future<void> _startStream() async {
    if (Platform.isMacOS) {
      await _startMacStream();
    } else {
      await _startMobileStream();
    }
  }

  Future<void> _stopStream() async {
    if (Platform.isMacOS) {
      await _stopMacStream();
    } else {
      await _stopMobileStream();
    }
  }

  @override
  void dispose() {
    _controller?.dispose();
    _worker.stop();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    if (!_isCameraInitialized && !Platform.isMacOS) {
      return const Scaffold(body: Center(child: CircularProgressIndicator(color: Colors.white)));
    }

    return Scaffold(
      body: Stack(
        fit: StackFit.expand,
        children: [
          // 1. Full Screen Camera Preview
          if (Platform.isMacOS)
            _buildMacCameraPreview()
          else if (_controller != null)
            mobile_camera.CameraPreview(_controller!),
          
          // 2. AI PIP Window
          if (_currentMode != AppMode.neutral)
            Positioned(
              bottom: 150,
              left: 0,
              right: 0,
              child: Center(
                child: Container(
                  width: 256,
                  height: 256,
                  decoration: BoxDecoration(
                    color: Colors.black45,
                    border: Border.all(color: Colors.greenAccent, width: 2),
                    boxShadow: const [BoxShadow(color: Colors.black26, blurRadius: 10)],
                  ),
                  child: _realtimeImage != null
                      ? Image.memory(_realtimeImage!, width: 256, height: 256, fit: BoxFit.contain)
                      : const Center(child: CircularProgressIndicator(color: Colors.greenAccent)),
                ),
              ),
            ),

          // 3. Status
          Positioned(
            top: 60,
            left: 20,
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text("NIGHT AI", style: GoogleFonts.outfit(color: Colors.white, fontSize: 24, fontWeight: FontWeight.bold)),
                Text("${_currentFps.toStringAsFixed(1)} FPS", style: const TextStyle(color: Colors.greenAccent, fontSize: 12)),
                if (_lastError != null) Text(_lastError!, style: const TextStyle(color: Colors.redAccent, fontSize: 10)),
              ],
            ),
          ),

          // 4. Mode Selector
          Positioned(
            bottom: 40,
            left: 0,
            right: 0,
            child: Center(
              child: Container(
                padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 12),
                decoration: BoxDecoration(
                  color: Colors.black.withAlpha(200),
                  borderRadius: BorderRadius.circular(30),
                ),
                child: Row(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    _buildModeButton(AppMode.neutral, "OFF"),
                    const SizedBox(width: 24),
                    _buildModeButton(AppMode.nightToDay, "NIGHT VISION"),
                  ],
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }

  // macOS-specific camera preview using CameraMacOSView
  Widget _buildMacCameraPreview() {
    return CameraMacOSView(
      key: _macCameraKey,
      fit: BoxFit.cover,
      cameraMode: CameraMacOSMode.photo,
      enableAudio: false,
      onCameraInizialized: (CameraMacOSController controller) {
        setState(() {
          _macController = controller;
          _lastError = "macOS Camera Ready.";
        });
      },
    );
  }

  Widget _buildModeButton(AppMode mode, String label) {
    final bool isSelected = _currentMode == mode;
    return GestureDetector(
      onTap: () => _onModeChanged(mode),
      child: Text(
        label,
        style: GoogleFonts.outfit(
          color: isSelected ? Colors.greenAccent : Colors.white30,
          fontWeight: isSelected ? FontWeight.bold : FontWeight.normal,
          fontSize: 14,
        ),
      ),
    );
  }
}
