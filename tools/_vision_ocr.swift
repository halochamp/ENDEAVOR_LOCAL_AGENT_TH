import Foundation
import Vision
import AppKit

let args = CommandLine.arguments

// --langs : print supported recognition languages (accurate level)
if args.count > 1 && args[1] == "--langs" {
    let req = VNRecognizeTextRequest()
    req.recognitionLevel = .accurate
    if let langs = try? req.supportedRecognitionLanguages() {
        print(langs.joined(separator: ", "))
    }
    exit(0)
}

guard args.count > 1,
      let img = NSImage(contentsOfFile: args[1]),
      let cg = img.cgImage(forProposedRect: nil, context: nil, hints: nil) else {
    print("ERR: cannot load \(args.count > 1 ? args[1] : "<no path>")")
    exit(1)
}

let req = VNRecognizeTextRequest { (request, _) in
    guard let obs = request.results as? [VNRecognizedTextObservation] else { return }
    let lines = obs.compactMap { $0.topCandidates(1).first?.string }
    print(lines.joined(separator: " / "))
}
req.recognitionLevel = .accurate
req.usesLanguageCorrection = true
req.recognitionLanguages = ["th-TH", "en-US"]

let handler = VNImageRequestHandler(cgImage: cg, options: [:])
do { try handler.perform([req]) } catch { print("ERR perform: \(error)") }
