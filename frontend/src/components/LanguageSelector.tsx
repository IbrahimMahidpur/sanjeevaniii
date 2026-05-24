import React from "react";

interface Props {
  value: string;
  onChange: (lang: string) => void;
}

const options: { code: string; label: string }[] = [
  { code: "en", label: "English" },
  { code: "hi", label: "Hindi" },
  { code: "es", label: "Spanish" },
  { code: "ar", label: "Arabic" },
  { code: "fr", label: "French" },
];

export const LanguageSelector: React.FC<Props> = ({ value, onChange }) => (
  <select
    className="bg-white dark:bg-gray-800 text-gray-800 dark:text-gray-200 border rounded p-1"
    value={value}
    onChange={(e) => onChange(e.target.value)}
  >
    {options.map((opt) => (
      <option key={opt.code} value={opt.code}>
        {opt.label}
      </option>
    ))}
  </select>
);
