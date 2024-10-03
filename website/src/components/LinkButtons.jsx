/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

import React from 'react';
import Link from '@docusaurus/Link';
import IconExternalLink from '@theme/Icon/ExternalLink'

const LinkButtons = ({githubUrl, colabUrl}) => {
  return (
    <div>
      <Link to={githubUrl} className="button button--outline button--primary margin--xs">
        Open in GitHub
        <IconExternalLink />
      </Link>
      <Link to={colabUrl} className="button button--outline button--primary margin--xs">
        Run in Google Colab
        <IconExternalLink />
      </Link>
      <br/><br/>
    </div>
  );
};

export default LinkButtons;
