"use client";

import React, { forwardRef } from "react";

type FormTextareaProps = {
  id: string;
  label: string;
  name?: string;
  required?: boolean;
  placeholder?: string;
  maxLength?: number;
  rows?: number;
  value: string;
  onChange: (e: React.ChangeEvent<HTMLTextAreaElement>) => void;
  error?: string;
};

const FormTextarea = forwardRef<HTMLTextAreaElement, FormTextareaProps>(
  (
    {
      id,
      label,
      name,
      required = false,
      placeholder,
      maxLength,
      rows = 4,
      value,
      onChange,
      error,
    },
    ref,
  ) => {
    return (
      <div>
        <label htmlFor={id} className="block text-sm font-medium text-gray-700">
          {label}
        </label>
        <textarea
          id={id}
          name={name || id}
          required={required}
          placeholder={placeholder}
          maxLength={maxLength}
          rows={rows}
          value={value}
          onChange={onChange}
          ref={ref}
          className={`mt-1 block w-full text-gray-800 rounded-md border px-3 py-2 text-sm shadow-sm focus:outline-none focus:ring-1
            ${error ? "border-red-500 focus:border-red-500 focus:ring-red-500" : "border-gray-300 focus:border-blue-500 focus:ring-blue-500"}`}
        />
        {maxLength && (
          <p className="mt-1 text-xs text-gray-700 text-right">
            {value.length} / {maxLength}
          </p>
        )}
        {error && <p className="text-sm text-red-600 mt-1">{error}</p>}
      </div>
    );
  },
);

FormTextarea.displayName = "FormTextarea"; // Required when using forwardRef

export default FormTextarea;
