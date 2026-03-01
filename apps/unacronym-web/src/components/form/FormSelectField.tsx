import {
  FieldValues,
  Path,
  RegisterOptions,
  UseFormRegister,
} from "react-hook-form";

type Option = {
  label: string;
  value: string;
};

type FormSelectFieldProps<TFormValues extends FieldValues> = {
  name: Path<TFormValues>;
  label: string;
  register: UseFormRegister<TFormValues>;
  options: Option[];
  error?: string;
  validationRules?: RegisterOptions<TFormValues, Path<TFormValues>>;
  autoComplete?: string;
};

export default function FormSelectField<TFormValues extends FieldValues>({
  name,
  label,
  register,
  options,
  error,
  validationRules,
  autoComplete,
}: FormSelectFieldProps<TFormValues>) {
  const borderColor = error
    ? "border-red-500 focus:border-red-500 focus:ring-red-500"
    : "border-gray-300 focus:border-blue-500 focus:ring-blue-500";

  return (
    <div className="mb-4">
      <label
        htmlFor={name}
        className="block text-sm font-medium text-gray-700 mb-1"
      >
        {label}
        {validationRules?.required && (
          <span className="text-red-500 font-bold"> *</span>
        )}
      </label>

      <div className="relative">
        <select
          id={name}
          {...register(name, validationRules)}
          autoComplete={autoComplete}
          className={`
            block w-full appearance-none rounded-md bg-white px-3 py-2 pr-10 text-sm shadow-sm
            focus:outline-none focus:ring-1
            disabled:opacity-50
            ${borderColor}
          `}
        >
          {options.map((opt) => (
            <option key={opt.value} value={opt.value}>
              {opt.label}
            </option>
          ))}
        </select>

        {/* Chevron Icon */}
        <div className="pointer-events-none absolute inset-y-0 right-2 flex items-center pr-2 text-gray-400">
          <svg
            className="h-4 w-4"
            viewBox="0 0 20 20"
            fill="currentColor"
            aria-hidden="true"
          >
            <path
              fillRule="evenodd"
              d="M10 12a1 1 0 0 1-.7-.3l-4-4a1 1 0 1 1 1.4-1.4L10 9.58l3.3-3.28a1 1 0 1 1 1.4 1.42l-4 4A1 1 0 0 1 10 12z"
              clipRule="evenodd"
            />
          </svg>
        </div>
      </div>

      {error && <p className="text-sm text-red-600 mt-1">{error}</p>}
    </div>
  );
}
