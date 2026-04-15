export type RegisterInterestFormValues = {
  name: string;
  email: string;
  phone: string;
  beta_opt_in: string;
  subject: string;
  area: string;
  terms: string;
  contact_consent: string;
  comments: string;
  id: string;
};

export type RegisterFormValues = {
  first_name: string;
  middle_name?: string;
  last_name: string;
  email: string;
  password: string;
  confirmPassword: string;
  terms: string;
  role: string;
  instructor_type: string;
  contact_consent: string;
  address: {
    street: string;
    local_area?: string;
    city: string;
    postcode: string;
    country: string;
  };
};

export type CompleteFormValues = {
  first_name: string;
  middle_name?: string;
  last_name: string;
  street: string;
  local_area?: string;
  city: string;
  postcode: string;
  country: string;
  role: string;
  instructor_type: string;
  terms: boolean;
  contact_consent: boolean;
};

export type EmailFormValues = {
  email: string;
};

export type ResetPWFormValues = {
  password: string;
  confirmPassword: string;
};

export type LoginFormValues = {
  email: string;
  password: string;
  staySignedIn: boolean;
};
