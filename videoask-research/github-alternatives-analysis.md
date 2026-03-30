# GitHub Repository Analysis: Open-Source VideoAsk Alternatives

This document provides a detailed analysis of open-source repositories that attempt to replicate or provide alternatives to VideoAsk. The analysis focuses on code structure, implemented features, technology stacks, quality assessments, and key takeaways that can inform the development of **VideoInquire**, a self-hosted, white-label alternative.

## 1. FormOnce (FormOnce/FormOnce)

FormOnce is the most comprehensive open-source alternative to VideoAsk currently available on GitHub. It positions itself as the "#1 open-source platform for building video enabled forms" and provides a full-stack solution for creating interactive, video-based surveys with conditional logic.

### Code Structure
The repository is structured as a modern monorepo using the T3 Stack. It utilizes Next.js for both the frontend and API routes, with Prisma handling database interactions. The codebase is well-organized into components, layouts, pages, and server routers. Notably, it includes a sophisticated `form-builder` directory containing a `flow-builder` sub-module that implements a visual node-based editor for conditional logic.

### Features Implemented
FormOnce implements a robust set of features that closely mirror VideoAsk's core functionality. It includes a visual flow builder for creating multi-step forms with branching logic. The platform supports video recording directly in the browser, video file uploads, and playback. It also features a comprehensive dashboard for managing forms, viewing responses, and configuring webhooks and API keys. The application supports multiple question types, including text, select, and video responses.

### Tech Stack Used
| Category | Technology |
| :--- | :--- |
| **Framework** | Next.js (React), T3 Stack |
| **Database & ORM** | PostgreSQL, Prisma |
| **API Layer** | tRPC |
| **Styling & UI** | Tailwind CSS, shadcn/ui |
| **Video Handling** | tus-js-client (resumable uploads), BunnyCDN (storage/streaming) |
| **Flow Builder** | React Flow |
| **Authentication** | NextAuth.js |

### Quality Assessment
The code quality is exceptionally high. It is a production-ready application with strict TypeScript typing, comprehensive schema validation using Zod, and a clean, modular architecture. The use of modern tools like tRPC ensures end-to-end type safety. The project is actively maintained and demonstrates professional software engineering practices.

### What Can Be Reused or Learned
FormOnce provides a masterclass in building a video form platform. The most valuable takeaway is their implementation of the visual flow builder using React Flow, which elegantly handles complex branching logic. Their approach to video handling—using `tus-js-client` for reliable chunked uploads and offloading processing/streaming to BunnyCDN—is highly effective, though for a fully self-hosted VideoInquire, this would need to be adapted to use MinIO and local FFmpeg processing. The database schema for forms, questions, and logic conditions is highly reusable.

## 2. fk-videoAsk (muhd-ameen/fk-videoAsk)

This repository is a modern, asynchronous video interview platform designed specifically for recruiters to create video questions and collect candidate responses.

### Code Structure
The project is a standard Vite + React single-page application (SPA). The `src` directory is cleanly divided into components, hooks, pages, and utility functions. It relies heavily on Supabase for its backend infrastructure, with SQL migration files included in the repository to define the database schema and Row Level Security (RLS) policies.

### Features Implemented
The application features a recruiter dashboard secured by email/password authentication. Recruiters can create interview flows by recording or uploading video questions. It generates unique, shareable links for candidates, who can then record their responses directly in the browser without needing to create an account. The platform handles video storage and retrieval seamlessly.

### Tech Stack Used
| Category | Technology |
| :--- | :--- |
| **Frontend** | React 18, TypeScript, Vite |
| **Styling** | Tailwind CSS, Lucide React (icons) |
| **Backend & Auth** | Supabase (PostgreSQL, Auth, Storage) |
| **Video Recording** | Native MediaRecorder API |
| **Testing** | Vitest, React Testing Library |

### Quality Assessment
The codebase is clean, modern, and well-structured. It utilizes custom React hooks (e.g., `useVideoRecorder`) to encapsulate complex logic, making the components highly readable. The inclusion of comprehensive tests and strict TypeScript configurations indicates a focus on reliability and maintainability.

### What Can Be Reused or Learned
The `useVideoRecorder` hook is an excellent, reusable implementation of the native MediaRecorder API, handling state management for recording time, video blobs, and preview streams. The Supabase schema provides a solid foundation for structuring questions and responses in a relational database. The approach to generating unique, unauthenticated links for candidates is a crucial pattern for VideoInquire.

## 3. videoAsk (GunkaArtur/videoAsk)

This is a very early-stage proof-of-concept attempting to replicate the front-end interaction model of the VideoAsk widget.

### Code Structure
The repository is a minimal Create React App project with only a handful of source files. It lacks a backend, routing, or state management library. The core logic is contained within `App.js`, `intro.jsx`, and dedicated files for audio and video recording.

### Features Implemented
The prototype successfully implements a video intro playback using a hardcoded URL. It provides a UI for users to select their preferred answer mode (video, audio, or text). It utilizes the MediaRecorder API to capture both video and audio, complete with basic play, stop, and redo controls. However, it lacks any backend integration, branching logic, or form persistence.

### Tech Stack Used
| Category | Technology |
| :--- | :--- |
| **Frontend** | React 16 (mixed class and functional components) |
| **Styling** | Plain CSS |
| **Audio Playback** | react-sound |
| **Media Capture** | Native MediaRecorder API, getUserMedia |

### Quality Assessment
The maturity of this project is very low. It serves purely as a frontend prototype. The code mixes older React class components with newer hooks, and relies on direct DOM manipulation in some areas. It has not been updated in several years.

### What Can Be Reused or Learned
While the code itself is not suitable for reuse, the UX pattern it attempts to replicate is valuable. The 360x630px card layout effectively mimics VideoAsk's mobile-first, embeddable widget design. It also demonstrates the basic usage of `getUserMedia` with `facingMode: "user"` to ensure the front-facing camera is used on mobile devices.

## 4. videoask_task_assignment (NayanDevLab/videoask_task_assignment)

This repository is a frontend clone focused on replicating the landing page and basic campaign UI of VideoAsk.

### Code Structure
Built with Create React App, the project structure is typical for a basic React application. It includes directories for assets, components, pages, and Redux slices.

### Features Implemented
The project primarily implements static UI components, including a home page, a campaign view, and a multi-step form UI. It features a custom `VideoPlayer` component with controls for playback speed, fullscreen, and custom overlays. It uses Redux Toolkit for basic state management across the UI steps.

### Tech Stack Used
| Category | Technology |
| :--- | :--- |
| **Frontend** | React 18 |
| **State Management** | Redux Toolkit |
| **Styling** | Tailwind CSS, AOS (Animate On Scroll) |

### Quality Assessment
This is a purely cosmetic clone. It does not implement actual video recording, backend communication, or dynamic form building. The code quality is adequate for a UI mockup but lacks the functional depth required for a working application.

### What Can Be Reused or Learned
The custom `VideoPlayer` component offers insights into building a tailored video playback experience with custom controls and overlays, which is essential for maintaining a white-label aesthetic in VideoInquire. The use of Tailwind CSS for rapid UI replication is also demonstrated effectively.

## 5. developstoday-videoask-task (yaroslavgorshkov/developstoday-videoask-task)

This repository is a collection of isolated React UI components built as a technical task, rather than a full application clone.

### Code Structure
The project is built using Next.js (App Router) and is heavily focused on component isolation and testing. It includes a Storybook setup for component documentation and Vitest for unit testing.

### Features Implemented
It implements several highly polished, reusable UI components, including a versatile `Input` component (handling text, password, and numbers with clearable states), a collapsible `SidebarMenu` with nested items, and a robust `Toast` notification system.

### Tech Stack Used
| Category | Technology |
| :--- | :--- |
| **Framework** | Next.js 16, React 19 |
| **Styling** | Tailwind CSS, Framer Motion, clsx, tailwind-merge |
| **Component Dev** | Storybook |
| **Testing** | Vitest, Playwright |

### Quality Assessment
The code quality for the individual components is excellent. The project demonstrates modern best practices, including the use of Storybook for component-driven development, comprehensive testing, and smooth animations using Framer Motion.

### What Can Be Reused or Learned
While it doesn't provide video-specific functionality, the repository is an excellent reference for building a high-quality, accessible, and animated UI component library. The implementation of the `SidebarMenu` and `Toast` components could be directly adapted for the VideoInquire administrative dashboard. The rigorous testing and Storybook setup represent an ideal development workflow.
