import { SignUp } from "@clerk/nextjs";
import type { Metadata } from "next";

import { AuthCardPage } from "@/components/auth-card-page";

export const metadata: Metadata = { title: "Register" };

export default function SignUpPage() {
  return (
    <AuthCardPage>
      <SignUp />
    </AuthCardPage>
  );
}