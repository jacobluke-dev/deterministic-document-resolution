import { FieldValues, Path, UseFormRegister } from "react-hook-form";
import CustomExternalLink from "@/components/customExternalLink/CustomExternalLink";

type TermsConsentCheckboxProps<TFormValues extends FieldValues> = {
  name: Path<TFormValues>;
  required?: boolean;
  register: UseFormRegister<TFormValues>;
  error?: string;
  requiredMessage?: string;
};

export default function TermsConsentCheckbox<TFormValues extends FieldValues>({
  name,
  required = false,
  register,
  error,
  requiredMessage,
}: TermsConsentCheckboxProps<TFormValues>) {
  return (
    <div className="text-sm text-gray-700">
      <div className="flex items-start gap-2">
        <input
          type="checkbox"
          id={name}
          {...register(name, {
            required: required
              ? requiredMessage || "You must agree to the terms."
              : false,
          })}
          className="mt-1"
        />
        <label htmlFor={name} className="leading-snug">
          I agree to the{" "}
          <CustomExternalLink href="/pages/privacy">
            Privacy Policy
          </CustomExternalLink>{" "}
          and{" "}
          <CustomExternalLink href="/pages/terms">
            Terms of Service
          </CustomExternalLink>
          .{required && <span className="text-red-500 font-bold"> *</span>}
        </label>
      </div>
      {error && <p className="text-sm text-red-600 mt-1">{error}</p>}
    </div>
  );
}
