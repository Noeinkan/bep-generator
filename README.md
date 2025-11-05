# BIM Execution Plan (BEP) Generator

A React-based application for generating comprehensive BIM Execution Plans (BEPs) in accordance with ISO 19650 standards. This tool helps project teams create detailed plans for BIM implementation, including Task Information Delivery Plans (TIDPs) and Master Information Delivery Plans (MIDPs).

## Features

- Interactive BEP creation wizard
- Support for both pre-appointment and post-appointment BEPs
- MIDP and TIDP planning and coordination
- Export capabilities (PDF, DOCX)
- User authentication and project management
- Real-time collaboration features
- **AI-Powered Text Generation** - ML-based content suggestions for BEP sections

## Documentation

For detailed information about BIM concepts and standards:

- [TIDP and MIDP Relationship](TIDP_MIDP_Relationship.md) - Understanding the relationship between Task Information Delivery Plans and Master Information Delivery Plans in ISO 19650 context.
- [AI Integration Guide](AI_INTEGRATION_GUIDE.md) - Complete guide for setting up and using the AI text generation feature.

## Getting Started

This project was bootstrapped with [Create React App](https://github.com/facebook/create-react-app).

### Prerequisites

- Node.js (v14 or higher)
- npm or yarn

### Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   npm install
   ```
3. **(Optional) Set up AI text generation:**
   ```bash
   setup-ai.bat
   ```
   This will install Python dependencies and train the AI model. See [AI Integration Guide](AI_INTEGRATION_GUIDE.md) for details.

4. Start the development server:
   ```bash
   npm start
   ```

   If using AI features, also start the ML service in a separate terminal:
   ```bash
   cd ml-service
   start_service.bat
   ```

### Available Scripts

In the project directory, you can run:

#### `npm start`

Runs the app in the development mode.\
Open [http://localhost:3000](http://localhost:3000) to view it in your browser.

The page will reload when you make changes.\
You may also see any lint errors in the console.

#### `npm test`

Launches the test runner in the interactive watch mode.\
See the section about [running tests](https://facebook.github.io/create-react-app/docs/running-tests) for more information.

#### `npm run build`

Builds the app for production to the `build` folder.\
It correctly bundles React in production mode and optimizes the build for the best performance.

The build is minified and the filenames include the hashes.\
Your app is ready to be deployed!

See the section about [deployment](https://facebook.github.io/create-react-app/docs/deployment) for more information.

#### `npm run eject`

**Note: this is a one-way operation. Once you `eject`, you can't go back!**

If you aren't satisfied with the build tool and configuration choices, you can `eject` at any time. This command will remove the single build dependency from your project.

Instead, it will copy all the configuration files and the transitive dependencies (webpack, Babel, ESLint, etc) right into your project so you have full control over them. All of the commands except `eject` will still work, but they will point to the copied scripts so you can tweak them. At this point you're on your own.

You don't have to ever use `eject`. The curated feature set is suitable for small and middle deployments, and you shouldn't feel obligated to use this feature. However we understand that this tool wouldn't be useful if you couldn't customize it when you are ready for it.

## Learn More

You can learn more in the [Create React App documentation](https://facebook.github.io/create-react-app/docs/getting-started).

To learn React, check out the [React documentation](https://reactjs.org/).

### Code Splitting

This section has moved here: [https://facebook.github.io/create-react-app/docs/code-splitting](https://facebook.github.io/create-react-app/docs/code-splitting)

### Analyzing the Bundle Size

This section has moved here: [https://facebook.github.io/create-react-app/docs/analyzing-the-bundle-size](https://facebook.github.io/create-react-app/docs/analyzing-the-bundle-size)

### Making a Progressive Web App

This section has moved here: [https://facebook.github.io/create-react-app/docs/making-a-progressive-web-app](https://facebook.github.io/create-react-app/docs/making-a-progressive-web-app)

### Advanced Configuration

This section has moved here: [https://facebook.github.io/create-react-app/docs/advanced-configuration](https://facebook.github.io/create-react-app/docs/advanced-configuration)

### Deployment

This section has moved here: [https://facebook.github.io/create-react-app/docs/deployment](https://facebook.github.io/create-react-app/docs/deployment)

### `npm run build` fails to minify

This section has moved here: [https://facebook.github.io/create-react-app/docs/troubleshooting#npm-run-build-fails-to-minify](https://facebook.github.io/create-react-app/docs/troubleshooting#npm-run-build-fails-to-minify)
