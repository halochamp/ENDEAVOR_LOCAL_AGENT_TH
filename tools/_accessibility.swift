// ENDEAVOR_LOCAL_AGENT_TH — © HaloChamp
// License: MIT License + Commons Clause — personal/educational use only, no commercial use without permission
// Website: https://www.poomwat.com | GitHub: https://github.com/halochamp | Email: champoomwat@gmail.com

import AppKit
import ApplicationServices
import Foundation

func emit(_ x:[String:Any])->Never {
    let d=(try? JSONSerialization.data(withJSONObject:x)) ?? Data("{\"status\":\"serialization_error\",\"elements\":[]}".utf8)
    print(String(data:d,encoding:.utf8)!)
    exit(0)
}

guard AXIsProcessTrusted() else { emit(["status":"permission_required","elements":[]]) }
guard let front=NSWorkspace.shared.frontmostApplication else { emit(["status":"no_frontmost_app","elements":[]]) }
let app=AXUIElementCreateApplication(front.processIdentifier)

func attr(_ e:AXUIElement,_ k:CFString)->CFTypeRef? {
    var v:CFTypeRef?
    return AXUIElementCopyAttributeValue(e,k,&v) == .success ? v:nil
}
func str(_ e:AXUIElement,_ k:CFString)->String? {
    (attr(e,k) as? String).flatMap{$0.isEmpty ? nil:$0}
}
func bool(_ e:AXUIElement,_ k:CFString)->Bool? {
    if let x=attr(e,k) as? Bool{return x}
    return (attr(e,k) as? NSNumber)?.boolValue
}
func point(_ e:AXUIElement,_ k:CFString)->CGPoint? {
    guard let r=attr(e,k),CFGetTypeID(r)==AXValueGetTypeID() else{return nil}
    var p=CGPoint.zero
    return AXValueGetValue(unsafeBitCast(r,to:AXValue.self),.cgPoint,&p) ? p:nil
}
func size(_ e:AXUIElement,_ k:CFString)->CGSize? {
    guard let r=attr(e,k),CFGetTypeID(r)==AXValueGetTypeID() else{return nil}
    var s=CGSize.zero
    return AXValueGetValue(unsafeBitCast(r,to:AXValue.self),.cgSize,&s) ? s:nil
}
func children(_ e:AXUIElement)->[AXUIElement] {
    attr(e,kAXChildrenAttribute as CFString) as? [AXUIElement] ?? []
}
func actions(_ e:AXUIElement)->[String] {
    var a:CFArray?
    return AXUIElementCopyActionNames(e,&a) == .success ? (a as? [String] ?? []):[]
}
func value(_ e:AXUIElement)->String? {
    if str(e,kAXRoleAttribute as CFString)=="AXSecureTextField" {return nil}
    if let v=attr(e,kAXValueAttribute as CFString) as? String{return String(v.prefix(240))}
    return (attr(e,kAXValueAttribute as CFString) as? NSNumber)?.stringValue
}

if CommandLine.arguments.dropFirst().first == "--insert-focused" {
    let inputData=FileHandle.standardInput.readDataToEndOfFile()
    guard let input=String(data:inputData,encoding:.utf8) else { emit(["status":"invalid_input"]) }
    guard let focusedRaw=attr(app,kAXFocusedUIElementAttribute as CFString) else { emit(["status":"focused_element_unavailable"]) }
    let focused=unsafeBitCast(focusedRaw,to:AXUIElement.self)
    let role=str(focused,kAXRoleAttribute as CFString) ?? ""
    if role == "AXSecureTextField" { emit(["status":"secure_field"]) }
    var settable:DarwinBoolean=false
    guard AXUIElementIsAttributeSettable(focused,kAXValueAttribute as CFString,&settable) == .success, settable.boolValue else {
        emit(["status":"value_not_settable","role":role])
    }
    guard let current=attr(focused,kAXValueAttribute as CFString) as? String else {
        emit(["status":"value_unavailable","role":role])
    }
    guard let rangeRaw=attr(focused,kAXSelectedTextRangeAttribute as CFString), CFGetTypeID(rangeRaw)==AXValueGetTypeID() else {
        emit(["status":"selection_unavailable","role":role])
    }
    var selection=CFRange()
    guard AXValueGetValue(unsafeBitCast(rangeRaw,to:AXValue.self),.cfRange,&selection) else {
        emit(["status":"selection_unavailable","role":role])
    }
    let currentNS=current as NSString
    guard selection.location >= 0, selection.length >= 0,
          selection.location <= currentNS.length,
          selection.location + selection.length <= currentNS.length else {
        emit(["status":"selection_invalid","role":role])
    }
    let replacementRange=NSRange(location:selection.location,length:selection.length)
    let updated=currentNS.replacingCharacters(in:replacementRange,with:input)
    guard AXUIElementSetAttributeValue(focused,kAXValueAttribute as CFString,updated as CFTypeRef) == .success else {
        emit(["status":"set_value_failed","role":role])
    }
    var caret=CFRange(location:selection.location + (input as NSString).length,length:0)
    if let caretValue=AXValueCreate(.cfRange,&caret) {
        _ = AXUIElementSetAttributeValue(focused,kAXSelectedTextRangeAttribute as CFString,caretValue)
    }
    emit(["status":"ok","role":role])
}

let limit=min(max(Int(CommandLine.arguments.dropFirst().first ?? "220") ?? 220,1),500)
let focused=attr(app,kAXFocusedWindowAttribute as CFString)
let root=focused.map{unsafeBitCast($0,to:AXUIElement.self)} ?? app
let title=str(root,kAXTitleAttribute as CFString) ?? ""
var wb:[Double]=[]
if let p=point(root,kAXPositionAttribute as CFString),let s=size(root,kAXSizeAttribute as CFString){
    wb=[Double(p.x),Double(p.y),Double(s.width),Double(s.height)]
}
let focusedUI=attr(app,kAXFocusedUIElementAttribute as CFString)
let focusedUIElem=focusedUI.map{unsafeBitCast($0,to:AXUIElement.self)}
let focusedUIHash:CFHashCode?=focusedUIElem.map{CFHash($0)}
var out:[[String:Any]]=[]
var seen=Set<CFHashCode>()
func walk(_ e:AXUIElement,_ depth:Int) {
    guard out.count<limit,depth<=10,seen.insert(CFHash(e)).inserted else{return}
    let role=str(e,kAXRoleAttribute as CFString) ?? "AXUnknown"
    let name=str(e,kAXTitleAttribute as CFString) ?? str(e,kAXDescriptionAttribute as CFString) ?? value(e) ?? ""
    let a=actions(e)
    if let p=point(e,kAXPositionAttribute as CFString),let s=size(e,kAXSizeAttribute as CFString),s.width>0,s.height>0,(!name.isEmpty || !a.isEmpty) {
        var r:[String:Any]=["role":role,"name":String(name.prefix(240)),"bounds":[Double(p.x),Double(p.y),Double(s.width),Double(s.height)],"actions":a]
        if let v=value(e),v != name {r["value"]=v}
        if let b=bool(e,kAXEnabledAttribute as CFString){r["enabled"]=b}
        let selfFocused=bool(e,kAXFocusedAttribute as CFString)
        let hashMatch=focusedUIHash != nil && CFHash(e)==focusedUIHash!
        if selfFocused==true || hashMatch {r["focused"]=true} else if let b=selfFocused {r["focused"]=b}
        out.append(r)
    }
    for c in children(e) where out.count<limit {walk(c,depth+1)}
}
walk(root,0)
emit(["status":"ok","app":front.localizedName ?? "","pid":front.processIdentifier,"window":title,"window_bounds":wb,"elements":out])
