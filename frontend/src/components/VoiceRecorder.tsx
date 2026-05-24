import React, { useState, useRef, useEffect } from "react";

interface Props {
  onTranscript: (text: string) => void;
}

const VoiceRecorder: React.FC<Props> = ({ onTranscript }) => {
  const [recording, setRecording] = useState(false);
  const [seconds, setSeconds] = useState(0);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const timerRef = useRef<NodeJS.Timeout>();

  useEffect(() => {
    return () => {
      // Cleanup on unmount
      if (mediaRecorderRef.current && mediaRecorderRef.current.state !== "inactive") {
        mediaRecorderRef.current.stop();
      }
      if (timerRef.current) clearInterval(timerRef.current);
    };
  }, []);

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mediaRecorder = new MediaRecorder(stream);
      mediaRecorderRef.current = mediaRecorder;
      chunksRef.current = [];

      mediaRecorder.ondataavailable = (e) => {
        if (e.data.size > 0) chunksRef.current.push(e.data);
      };

      mediaRecorder.onstop = async () => {
        const blob = new Blob(chunksRef.current, { type: "audio/webm" });
        const reader = new FileReader();
        reader.onloadend = async () => {
          const base64 = (reader.result as string).split(",")[1];
          // Send to backend for transcription – using the same chat endpoint for simplicity.
          try {
            const response = await fetch("/api/chat", {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({
                message: "",
                session_id: "temp-voice-session",
                file: base64,
                filename: "voice.webm",
              }),
            });
            const json = await response.json();
            if (json && json.response) {
              onTranscript(json.response);
            }
          } catch (err) {
            console.error("Voice transcription failed", err);
          }
        };
        reader.readAsDataURL(blob);
      };

      mediaRecorder.start();
      setRecording(true);
      setSeconds(0);
      timerRef.current = setInterval(() => setSeconds((s) => s + 1), 1000);
    } catch (err) {
      console.error("Failed to start recording", err);
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== "inactive") {
      mediaRecorderRef.current.stop();
    }
    setRecording(false);
    if (timerRef.current) clearInterval(timerRef.current);
  };

  return (
    <button
      type="button"
      className="px-3 py-1 bg-green-600 text-white rounded-md disabled:opacity-50"
      onClick={recording ? stopRecording : startRecording}
    >
      {recording ? `Stop (${seconds}s)` : "Record Voice"}
    </button>
  );
};

export default VoiceRecorder;
