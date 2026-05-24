import React from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import rehypeRaw from "rehype-raw";

export type Message = {
  role: "user" | "assistant";
  content: string;
  intent?: string; // optional intent for assistant messages
};

const intentBadge = (intent?: string) => {
  if (!intent) return null;
  const map: Record<string, { color: string; label: string }> = {
    emergency: { color: "bg-red-600 text-white", label: "EMERGENCY" },
    drug_info: { color: "bg-blue-600 text-white", label: "Drug Info" },
    lab_report: { color: "bg-purple-600 text-white", label: "Lab Report" },
    symptom_check: { color: "bg-green-600 text-white", label: "Symptom Check" },
    general: { color: "bg-gray-600 text-white", label: "General" },
  };
  const info = map[intent] || { color: "bg-gray-500 text-white", label: intent };
  return (
    <span className={`ml-2 px-2 py-0.5 rounded text-xs ${info.color}`}> {info.label} </span>
  );
};

export const MessageBubble: React.FC<{ message: Message }> = ({ message }) => {
  const isUser = message.role === "user";
  
  if (isUser) {
    return (
      <div className="flex justify-end my-4">
        <div className="max-w-[70%] rounded-3xl px-5 py-2.5 bg-[#2f2f2f] text-gray-200">
          <ReactMarkdown remarkPlugins={[remarkGfm]} rehypePlugins={[rehypeRaw]}>{message.content}</ReactMarkdown>
        </div>
      </div>
    );
  }

  // Assistant message
  return (
    <div className="flex justify-start my-6">
      <div className="w-full text-gray-200 prose prose-invert max-w-none">
        <ReactMarkdown remarkPlugins={[remarkGfm]} rehypePlugins={[rehypeRaw]}>{message.content}</ReactMarkdown>
        {message.intent && intentBadge(message.intent)}
      </div>
    </div>
  );
};
