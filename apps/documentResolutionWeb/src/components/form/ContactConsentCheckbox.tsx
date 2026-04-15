import { FieldValues, Path, UseFormRegister } from "react-hook-form";

type ContactConsentCheckboxProps<TFormValues extends FieldValues> = {
  name: Path<TFormValues>;
  label: string;
  required?: boolean;
  register: UseFormRegister<TFormValues>;
  error?: string;
  requiredMessage?: string;
};

export default function ContactConsentCheckbox<
  TFormValues extends FieldValues,
>({
  name,
  label,
  required = false,
  register,
  error,
  requiredMessage,
}: ContactConsentCheckboxProps<TFormValues>) {
  return (
    <div className="text-sm text-gray-700">
      <div className="flex items-start gap-2">
        <input
          type="checkbox"
          id={name}
          {...register(name, {
            required: required
              ? requiredMessage ||
                "Please provide your consent to be contacted."
              : false,
          })}
          className="mt-1"
        />
        <label
          htmlFor={name}
          className="leading-snug"
          dangerouslySetInnerHTML={{
            __html: `${label}${required ? '<span class="text-red-500 font-bold"> *</span>' : ""}`,
          }}
        />
      </div>
      {error && <p className="text-sm text-red-600 mt-1">{error}</p>}
    </div>
  );
}
