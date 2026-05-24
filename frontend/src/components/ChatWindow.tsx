import React, { useState, useEffect, useRef } from "react";
import { MessageBubble, Message } from "./MessageBubble";
import { useSSEStream } from "../hooks/useSSEStream";
import { v4 as uuidv4 } from "uuid";
import { Plus, Mic, ArrowUp, Paperclip, X } from "lucide-react";

interface Props {
  language: string;
}

const ChatWindow: React.FC<Props> = ({ language }) => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState<string>("");
  const [sessionId] = useState<string>(() => uuidv4());
  const [fileData, setFileData] = useState<{ b64: string; name: string } | null>(
    null
  );
  const [isStreaming, setIsStreaming] = useState<boolean>(false);
  const [streamParams, setStreamParams] = useState<
    Record<string, string | undefined> | null
  >(null);
  const [showAttachMenu, setShowAttachMenu] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const bottomRef = useRef<HTMLDivElement>(null);

  // Close attach menu when clicking outside
  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      if (!(e.target as Element).closest('.attach-menu-container')) {
        setShowAttachMenu(false);
      }
    };
    document.addEventListener('click', handleClickOutside);
    return () => document.removeEventListener('click', handleClickOutside);
  }, []);

  const { data: streamData, error, done } = useSSEStream(
    "http://localhost:8080/chat/stream",
    streamParams
  );

  useEffect(() => {
    if (!isStreaming) return;
    if (streamData) {
      setMessages((prev) => {
        const last = prev[prev.length - 1];
        if (last && last.role === "assistant") {
          const updated = { ...last, content: streamData };
          return [...prev.slice(0, -1), updated];
        }
        return prev;
      });
    }
    if (done) {
      setIsStreaming(false);
      setStreamParams(null);
    }
  }, [streamData, done, isStreaming]);

  const handleSend = async () => {
    if (!input.trim() && !fileData) return;

    const userMsg: Message = {
      role: "user",
      content: fileData ? `${input} [File: ${fileData.name}]` : input,
    };
    setMessages((m) => [...m, userMsg]);
    setMessages((m) => [...m, { role: "assistant", content: "" }]);
    setIsStreaming(true);

    const currentInput = input;
    const currentFile = fileData;
    setInput("");
    setFileData(null);

    if (currentFile) {
      // File upload — use POST endpoint
      try {
        const res = await fetch("http://localhost:8080/chat", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            "Accept-Language": language,
          },
          body: JSON.stringify({
            message: currentInput || "Please analyze this file",
            session_id: sessionId,
            file: currentFile.b64,
            filename: currentFile.name,
          }),
        });

        if (!res.ok) {
          const errText = await res.text();
          console.error("POST error:", errText);
          setMessages((m) => {
            const last = m[m.length - 1];
            if (last?.role === "assistant") {
              return [...m.slice(0, -1), { role: "assistant", content: `Error: ${errText}` }];
            }
            return m;
          });
          setIsStreaming(false);
          return;
        }

        const result = await res.json();
        setMessages((m) => {
          const last = m[m.length - 1];
          if (last?.role === "assistant") {
            return [...m.slice(0, -1), { role: "assistant", content: result.response }];
          }
          return m;
        });
      } catch (e: any) {
        setMessages((m) => {
          const last = m[m.length - 1];
          if (last?.role === "assistant") {
            return [...m.slice(0, -1), { role: "assistant", content: `Error: ${e.message}` }];
          }
          return m;
        });
      }
      setIsStreaming(false);
    } else {
      // Text only — use SSE streaming
      setStreamParams({
        message: currentInput,
        session_id: sessionId,
      });
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isStreaming]);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (!files || files.length === 0) return;
    const file = files[0];
    const reader = new FileReader();
    reader.onload = () => {
      const result = reader.result as string;
      const b64 = result.split(",")[1];
      setFileData({ b64, name: file.name });
      setShowAttachMenu(false);
    };
    reader.readAsDataURL(file);
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  const renderInputArea = () => (
    <div className="w-full relative">
      {fileData && (
        <div className="absolute bottom-full mb-3 left-0 bg-[#2f2f2f] border border-[#444] rounded-xl p-3 flex items-center gap-3 shadow-lg">
          <div className="w-10 h-10 rounded bg-[#444] flex items-center justify-center">
            <Paperclip className="text-gray-300" size={20} />
          </div>
          <div className="flex flex-col">
            <span className="text-sm font-medium text-gray-200">{fileData.name}</span>
            <span className="text-xs text-gray-500">File attached</span>
          </div>
          <button
            onClick={() => setFileData(null)}
            className="ml-4 p-1 text-gray-400 hover:text-white hover:bg-gray-700 rounded-full"
          >
            <X size={16} />
          </button>
        </div>
      )}
      <div className="relative bg-[#2f2f2f] rounded-3xl p-2 flex items-end border border-[#444] focus-within:border-gray-500 transition-colors shadow-sm">
        <div className="relative attach-menu-container flex items-end">
          <button
            className="p-2 mb-1 text-gray-400 hover:text-white rounded-full hover:bg-gray-700 transition-colors"
            onClick={() => setShowAttachMenu(!showAttachMenu)}
          >
            <Plus size={20} />
          </button>
          {showAttachMenu && (
            <div className="absolute bottom-full left-0 mb-2 bg-[#2f2f2f] border border-[#444] rounded-xl shadow-xl w-48 overflow-hidden z-50">
              <button
                className="w-full flex items-center gap-3 px-4 py-3 text-sm text-gray-200 hover:bg-[#3f3f3f] transition-colors"
                onClick={() => fileInputRef.current?.click()}
              >
                <Paperclip size={18} />
                <span>Add photos & files</span>
              </button>
            </div>
          )}
        </div>
        <input
          type="file"
          className="hidden"
          ref={fileInputRef}
          accept="image/*,application/pdf"
          onChange={handleFileChange}
        />
        <textarea
          className="flex-1 bg-transparent border-none focus:outline-none text-white px-2 py-3 resize-none max-h-48 overflow-y-auto"
          rows={1}
          placeholder="Ask anything"
          value={input}
          onChange={(e) => {
            setInput(e.target.value);
            e.target.style.height = 'auto';
            e.target.style.height = e.target.scrollHeight + 'px';
          }}
          onKeyDown={handleKeyDown}
          disabled={isStreaming}
        />
        <div className="flex items-end">
          <button className="p-2 mb-1 text-gray-400 hover:text-white rounded-full hover:bg-gray-700 transition-colors mx-1">
            <Mic size={20} />
          </button>
          <button
            onClick={handleSend}
            disabled={(!input.trim() && !fileData) || isStreaming}
            className={`p-2 mb-1 rounded-full transition-colors ${input.trim() || fileData ? 'bg-white text-black hover:bg-gray-200' : 'bg-[#444] text-gray-500'}`}
          >
            <ArrowUp size={20} />
          </button>
        </div>
      </div>
      {error && <p className="text-red-600 text-center mt-2">{error}</p>}
    </div>
  );

  const isEmpty = messages.length === 0;

  return (
    <div className="flex flex-col h-full relative">
      {isEmpty ? (
        <div className="flex-1 flex flex-col items-center justify-center p-4">
          <h2 className="text-3xl font-semibold mb-8 text-white">
            How can I help you today?
          </h2>
          <div className="w-full max-w-3xl">
            {renderInputArea()}
          </div>
        </div>
      ) : (
        <>
          <div className="flex-1 overflow-y-auto w-full">
            <div className="max-w-4xl mx-auto p-4 w-full pb-40">
              {messages.map((msg, idx) => (
                <MessageBubble key={idx} message={msg} />
              ))}
              {isStreaming && (
                <MessageBubble message={{ role: "assistant", content: "…" }} />
              )}
              <div ref={bottomRef} />
            </div>
          </div>
          <div className="absolute bottom-0 left-0 w-full bg-gradient-to-t from-[#212121] via-[#212121] to-transparent pt-12 pb-4 px-4">
            <div className="max-w-3xl mx-auto">
              {renderInputArea()}
            </div>
          </div>
        </>
      )}
    </div>
  );
};

export default ChatWindow;
