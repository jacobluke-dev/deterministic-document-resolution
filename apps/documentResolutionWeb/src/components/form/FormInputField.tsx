import {
  FieldValues,
  Path,
  RegisterOptions,
  UseFormRegister,
} from "react-hook-form";

type FormInputFieldProps<TFormValues extends FieldValues> = {
  name: Path<TFormValues>;
  label: string;
  type?: string;
  required?: boolean;
  requiredMessage?: string;
  placeholder?: string;
  register: UseFormRegister<TFormValues>;
  error?: string;
  validationRules?: RegisterOptions<TFormValues, Path<TFormValues>>;
  autoComplete?: string;
};

export default function FormInputField<TFormValues extends FieldValues>({
  label,
  name,
  type = "text",
  placeholder,
  register,
  error,
  validationRules,
  autoComplete,
}: FormInputFieldProps<TFormValues>) {
  const isCheckbox = type === "checkbox";

  return (
    <div className={isCheckbox ? "flex items-start gap-2" : "mb-4"}>
      {isCheckbox ? (
        <>
          <input
            id={name}
            type="checkbox"
            {...register(name, validationRules)}
            className="mt-1"
            autoComplete={autoComplete}
          />
          <label htmlFor={name} className="text-sm text-gray-700 leading-snug">
            {label}
            {validationRules?.required && (
              <span className="text-red-500 font-bold"> *</span>
            )}
          </label>
        </>
      ) : (
        <>
          <label
            htmlFor={name}
            className="block text-sm font-medium text-gray-700"
          >
            {label}
            {validationRules?.required && (
              <span className="text-red-500 font-bold"> *</span>
            )}
          </label>
          <input
            id={name}
            {...register(name, validationRules)}
            type={type}
            placeholder={placeholder}
            autoComplete={autoComplete}
            className={`mt-1 block w-full rounded-md border px-3 py-2 text-sm shadow-sm
                        focus:outline-none focus:ring-1
                        ${
                          error
                            ? "border-red-500 focus:border-red-500 focus:ring-red-500"
                            : "border-gray-300 focus:border-blue-500 focus:ring-blue-500"
                        }`}
          />
        </>
      )}
      {error && <p className="text-sm text-red-600 mt-1">{error}</p>}
    </div>
  );
}
